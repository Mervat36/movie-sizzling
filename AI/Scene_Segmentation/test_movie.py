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
import argparse
import sys

# === Config ===
# Default values - can be overridden by command line arguments
DEFAULT_MOVIE_FOLDER = "heross22"  # Updated to use heross22 as default
DEFAULT_MODEL_PATH = "./best_model.pt"  # Updated to use the model in current directory
OUTPUT_SCENES_JSON = "scene_groups.json"
SCENE_SHEETS_DIR = "scene_sheets"
BATCH_SIZE = 1
SHOT_NUM = 8
SIM_CHANNEL = 128
LSTM_HIDDEN_SIZE = 128
BIDIRECTIONAL = True
CONFIDENCE_THRESHOLD = 0.2  # Optimized from 0.5
MIN_BOUNDARY_GAP = 2  # Optimized from 3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Movie Scene Segmentation Testing')
parser.add_argument('--movie_folder', type=str, default=DEFAULT_MOVIE_FOLDER,
                    help=f'Path to movie folder containing shot images (default: {DEFAULT_MOVIE_FOLDER})')
parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                    help=f'Path to trained model (default: {DEFAULT_MODEL_PATH})')
parser.add_argument('--output_json', type=str, default=OUTPUT_SCENES_JSON,
                    help=f'Output JSON file for scene groups (default: {OUTPUT_SCENES_JSON})')
parser.add_argument('--scene_sheets_dir', type=str, default=SCENE_SHEETS_DIR,
                    help=f'Directory to save scene contact sheets (default: {SCENE_SHEETS_DIR})')
parser.add_argument('--confidence_threshold', type=float, default=CONFIDENCE_THRESHOLD,
                    help=f'Confidence threshold for boundary detection (default: {CONFIDENCE_THRESHOLD})')
parser.add_argument('--min_boundary_gap', type=int, default=MIN_BOUNDARY_GAP,
                    help=f'Minimum gap between boundaries in shots (default: {MIN_BOUNDARY_GAP})')

args = parser.parse_args()

# Use command line arguments or defaults
MOVIE_FOLDER = args.movie_folder
MODEL_PATH = args.model_path
OUTPUT_SCENES_JSON = args.output_json
SCENE_SHEETS_DIR = args.scene_sheets_dir
CONFIDENCE_THRESHOLD = args.confidence_threshold
MIN_BOUNDARY_GAP = args.min_boundary_gap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print configuration
print(f"🎬 Movie folder: {MOVIE_FOLDER}")
print(f"🤖 Model path: {MODEL_PATH}")
print(f"📊 Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"🔧 Min boundary gap: {MIN_BOUNDARY_GAP}")
print(f"💻 Device: {device}")

# Validate inputs
if not os.path.exists(MOVIE_FOLDER):
    print(f"❌ Error: Movie folder '{MOVIE_FOLDER}' does not exist!")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' does not exist!")
    sys.exit(1)

# Check if movie folder contains shot files
shot_files = [f for f in os.listdir(MOVIE_FOLDER) if f.startswith("shot_")]
if not shot_files:
    print(f"❌ Error: No shot files found in '{MOVIE_FOLDER}'!")
    print("   Expected files with format: shot_XXXX_img_Y.jpg")
    sys.exit(1)

print(f"📁 Found {len(shot_files)} shot files in {MOVIE_FOLDER}")

# === Dataset ===
class MovieSceneTestDataset(Dataset):
    def __init__(self, folder):
        self.folder   = folder
        # Updated to work with actual file format: shot_XXX_start.jpg, shot_XXX_middle.jpg, shot_XXX_end.jpg
        # Only include shots that have ALL 3 files (start, middle, end)
        all_files = [f for f in os.listdir(folder) if f.startswith("shot_")]
        
        # Find complete shots (have all 3 files)
        complete_shots = set()
        for file in all_files:
            parts = file.split("_")
            if len(parts) >= 3:
                try:
                    shot_id = int(parts[1])
                    frame_type = parts[2].replace(".jpg", "")
                    if frame_type in ["start", "middle", "end"]:
                        # Check if this shot has all 3 files
                        start_file = os.path.join(folder, f"shot_{shot_id:03d}_start.jpg")
                        middle_file = os.path.join(folder, f"shot_{shot_id:03d}_middle.jpg")
                        end_file = os.path.join(folder, f"shot_{shot_id:03d}_end.jpg")
                        
                        if all(os.path.exists(f) for f in [start_file, middle_file, end_file]):
                            complete_shots.add(shot_id)
                except ValueError:
                    continue
        
        self.shot_ids = sorted(complete_shots)
        
        # Count total shots for comparison
        total_shots = len(set(int(f.split('_')[1]) for f in all_files 
                            if f.startswith('shot_') and len(f.split('_')) >= 2 
                            and f.split('_')[1].isdigit()))
        
        print(f"📊 Using {len(self.shot_ids)} complete shots (out of {total_shots} total)")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.shot_ids) - SHOT_NUM + 1

    def __getitem__(self, idx):
        """
        Returns a tensor of shape [SHOT_NUM, 3, H, W]:
          each shot reduced to the mean of its start, middle, and end frames.
        """
        seq = []
        for s in range(idx, idx + SHOT_NUM):
            frames = []
            shot_id = self.shot_ids[s]
            
            # Load start, middle, and end frames for each shot
            frame_types = ["start", "middle", "end"]
            for frame_type in frame_types:
                path = os.path.join(self.folder, f"shot_{shot_id:03d}_{frame_type}.jpg")
                # Since we only use complete shots, this should always exist
                img = Image.open(path).convert("RGB")
                frames.append(self.transform(img))
            
            # mean over the 3 frames (start, middle, end)
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
            # Updated to use the actual file format: shot_XXX_start.jpg
            path = os.path.join(folder, f"shot_{sid:03d}_start.jpg")
            try:
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB").resize(thumb_size)
                    thumbs.append(img)
                else:
                    print(f"⚠️  Warning: File not found for scene sheet: {path}")
                    # Create a dummy thumbnail
                    dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
                    thumbs.append(dummy_img)
            except Exception as e:
                print(f"❌ Error loading image for scene sheet {path}: {e}")
                # Create a dummy thumbnail
                dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
                thumbs.append(dummy_img)
        
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

    # Print scene shot ranges
    for idx, scene in enumerate(scenes, start=1):
        if scene:
            print(f"Scene {idx}: shots {scene[0]} to {scene[-1]}")

    # save JSON
    with open(OUTPUT_SCENES_JSON, "w") as f:
        json.dump(scenes, f, indent=2)
    print(f"💾 Saved JSON: {OUTPUT_SCENES_JSON}")

    # save one thumbnail per shot, per scene
    save_scene_sheets(scenes, MOVIE_FOLDER, SCENE_SHEETS_DIR)

    print("🎉 All done!")
