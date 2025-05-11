#!/usr/bin/env python3
import os
import json
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# === Config ===
MOVIE_FOLDER         = "D:/extra/tt0100157/tt0100157"
MODEL_PATH           = "./checkpoints/best_model.pt"
OUTPUT_SCENES_JSON   = "scene_groups.json"
SCENE_SHEETS_DIR     = "scene_sheets"
BATCH_SIZE           = 1
SHOT_NUM             = 8
SIM_CHANNEL          = 128
LSTM_HIDDEN_SIZE     = 128
BIDIRECTIONAL        = True
CONFIDENCE_THRESHOLD = 0.5
MIN_BOUNDARY_GAP     = 3  # shots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Dataset ===
class MovieSceneTestDataset(Dataset):
    def __init__(self, folder):
        self.folder   = folder
        # assume shot_<id>_img_<f>.jpg, collect unique IDs
        self.shot_ids = sorted({
            int(f.split("_")[1])
            for f in os.listdir(folder) if f.startswith("shot_")
        })
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.shot_ids) - SHOT_NUM + 1

    def __getitem__(self, idx):
        """
        Returns a tensor of shape [SHOT_NUM, 3, H, W]:
          each shot reduced to the mean of its first 3 frames.
        """
        seq = []
        for s in range(idx, idx + SHOT_NUM):
            frames = []
            for fno in range(3):
                path = os.path.join(self.folder,
                    f"shot_{self.shot_ids[s]:04d}_img_{fno}.jpg")
                img = Image.open(path).convert("RGB")
                frames.append(self.transform(img))
            # mean over the 3 frames
            seq.append(torch.stack(frames).mean(0))
        return torch.stack(seq)  # [SHOT_NUM, 3, H, W]


# === Model ===
class ShotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.dim      = resnet.fc.in_features

    def forward(self, x):
        # x: [B, S, C, H, W]
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        out = self.features(x).view(B, S, -1)  # [B, S, D]
        return out


class CosSimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear((SHOT_NUM//2)*dim, SIM_CHANNEL)

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        half = SHOT_NUM // 2
        p1, p2 = x[:, :half, :], x[:, half:, :]
        v1 = self.lin(p1.reshape(B, -1))
        v2 = self.lin(p2.reshape(B, -1))
        return F.cosine_similarity(v1, v2, dim=1)  # [B]


class LGSSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc   = ShotEncoder()
        self.cs    = CosSimBlock(self.enc.dim)
        lstm_layers = 2 if BIDIRECTIONAL else 1
        self.lstm  = nn.LSTM(
            self.enc.dim + SIM_CHANNEL,
            LSTM_HIDDEN_SIZE,
            num_layers=1,
            bidirectional=BIDIRECTIONAL,
            batch_first=True
        )
        out_dim = LSTM_HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        self.fc    = nn.Sequential(
            nn.Linear(out_dim, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        # x: [B, S, C, H, W]
        feats = self.enc(x)                   # [B, S, D]
        sim   = self.cs(feats)                # [B]
        sim   = sim.unsqueeze(1).unsqueeze(2) # [B,1,1]
        sim   = sim.expand(-1, SHOT_NUM - 1, SIM_CHANNEL)  # [B, S-1, SIM_CHANNEL]
        cat   = torch.cat([feats[:, :SHOT_NUM-1, :], sim], dim=2)
        out, _= self.lstm(cat)                # [B, S-1, hidden]
        center = SHOT_NUM // 2 - 1
        return self.fc(out[:, center, :])     # [B,2]


# === Prediction ===
def predict_scenes(model, loader, shot_ids):
    model.eval()
    confidences = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            # batch: [B=1, S, C, H, W] → remove batch dim
            x = batch.to(device)  # [1, S, C, H, W]
            logits = model(x)     # [1,2]
            prob_boundary = F.softmax(logits, dim=1)[0,1].item()
            # map center‐index → actual shot_id
            center_idx = idx + SHOT_NUM//2
            sid = shot_ids[center_idx]
            confidences.append((sid, prob_boundary))

    # threshold & peak‐pick
    raw = [(s,p) for s,p in confidences if p >= CONFIDENCE_THRESHOLD]
    peaks = []
    for i, (sid, cf) in enumerate(raw):
        prev_cf = raw[i-1][1] if i>0            else -1
        next_cf = raw[i+1][1] if i<len(raw)-1   else -1
        if cf >= prev_cf and cf >= next_cf:
            peaks.append(sid)
    # enforce minimum gap
    bounds = []
    for s in sorted(peaks):
        if bounds and shot_ids.index(s) - shot_ids.index(bounds[-1]) < MIN_BOUNDARY_GAP:
            continue
        bounds.append(s)
    print(f"🔖 Boundaries (shot IDs): {bounds}")
    return bounds


# === Grouping ===
def group_scenes(shot_ids, boundaries):
    """
    initial split at each boundary shot_id → scenes.
    then, merge any scenes of length 1 into the previous scene.
    """
    scenes = []
    curr = []
    bset = set(boundaries)
    for sid in shot_ids:
        if sid in bset and curr:
            scenes.append(curr)
            curr = []
        curr.append(sid)
    if curr:
        scenes.append(curr)

    # merge singletons
    merged = []
    for grp in scenes:
        if len(grp) == 1:
            if merged:
                merged[-1].extend(grp)
            else:
                # if first scene is singleton, merge into next
                # we'll handle by postponing
                continue
        else:
            merged.append(grp)
    # edge‐case: first was singleton
    if scenes and len(scenes[0]) == 1 and merged:
        merged[0] = scenes[0] + merged[0]
    return merged


# === Save contact sheets ===
def save_scene_sheets(scenes, folder, output_dir, thumb_size=(224,224)):
    os.makedirs(output_dir, exist_ok=True)
    for idx, scene in enumerate(scenes, start=1):
        thumbs = []
        for sid in scene:
            path = os.path.join(folder, f"shot_{sid:04d}_img_0.jpg")
            img  = Image.open(path).convert("RGB").resize(thumb_size)
            thumbs.append(img)
        # compose horizontal sheet
        w, h = thumb_size
        sheet = Image.new("RGB", (w*len(thumbs), h))
        for i, thumb in enumerate(thumbs):
            sheet.paste(thumb, (i*w, 0))
        out = os.path.join(output_dir, f"scene_{idx:03d}.png")
        sheet.save(out)
        print(f"💾 Saved scene {idx:03d} ({len(thumbs)} shots): {out}")


# === Main ===
if __name__ == "__main__":
    ds     = MovieSceneTestDataset(MOVIE_FOLDER)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    shot_ids = ds.shot_ids

    # load model
    model = LGSSModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("✅ Model loaded")

    # predict boundaries
    boundaries = predict_scenes(model, loader, shot_ids)

    # group into scenes
    scenes = group_scenes(shot_ids, boundaries)
    print(f"✅ {len(scenes)} scenes detected")

    # save JSON
    with open(OUTPUT_SCENES_JSON, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"💾 Saved JSON: {OUTPUT_SCENES_JSON}")

    # save one thumbnail per shot, per scene
    save_scene_sheets(scenes, MOVIE_FOLDER, SCENE_SHEETS_DIR)

    print("🎉 All done!")
