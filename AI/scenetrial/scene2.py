# %%
import json, os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, models

# Allow PIL to load broken/truncated JPEGs without hanging
ImageFile.LOAD_TRUNCATED_IMAGES = True

root_dir = "C:/Users/dell/Desktop/movies"
labels_file = "C:/Users/dell/Desktop/scenelabels.json"
features_dir = "C:/Users/dell/Desktop/good/features"
os.makedirs(features_dir, exist_ok=True)

# Load ResNet50 as feature extractor
cnn = models.resnet50(pretrained=True)
cnn.fc = nn.Identity()
cnn.eval().cuda()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

# Load scene labels
with open(labels_file) as f:
    meta = json.load(f)

folders = {p.name for p in Path(root_dir).iterdir() if p.is_dir()}

entries = []
if isinstance(meta, dict):
    for split in ['train','val','test']:
        if split in meta and isinstance(meta[split], list):
            entries.extend(meta[split])
elif isinstance(meta, list):
    entries = meta
else:
    raise ValueError("scene_labels.json must be dict or list")

# Build shot label map
shot_label_map = {}
for entry in entries:
    if not isinstance(entry, dict):
        continue
    mid = entry.get('name') or entry.get('movie_id')
    if not mid or mid not in folders:
        continue
    label_list = entry.get('label')
    if not isinstance(label_list, list):
        continue
    shot_label_map[mid] = [(sid, int(lbl)) for sid, lbl in label_list if lbl in ('0','1')]

print(f"Found {len(shot_label_map)} valid movies.")

# Precompute features
for mid, shots in tqdm(shot_label_map.items(), desc="Movies"):
    save_dir = Path(features_dir)/mid
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Starting movie {mid} with {len(shots)} shots")
    for sid, lbl in tqdm(shots, desc=f"Shots {mid}", leave=True):
        feat_file = save_dir/f"{sid}.pt"
        if feat_file.exists():
            continue  # already computed
        
        frames = []
        for i in range(3):
            img_path = Path(root_dir)/mid/f"shot_{sid}_img_{i}.jpg"
            print(f"[DEBUG] Movie {mid} Shot {sid} Frame {i} → loading {img_path}")
            if img_path.exists():
                try:
                    img = Image.open(str(img_path)).convert("RGB")
                    frames.append(transform(img))
                except Exception as e:
                    print(f"[WARNING] Skipping frame {img_path}: {e}")
            else:
                print(f"[WARNING] Frame file missing: {img_path}")
        
        if frames:
            frames = torch.stack(frames).cuda()
            with torch.no_grad():
                feats = cnn(frames)
            shot_feat = feats.mean(0)
            print(f"[INFO] Movie {mid} Shot {sid} → computed from {len(frames)} frames")
        else:
            shot_feat = torch.zeros(2048).cuda()
            print(f"[WARNING] Movie {mid} Shot {sid} → no valid frames → saving zero vector")
        
        torch.save(shot_feat.cpu(), feat_file)


# %%
import json, os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, models

ImageFile.LOAD_TRUNCATED_IMAGES = True

root_dir = "C:/Users/dell/Desktop/movies"
labels_file = "C:/Users/dell/Desktop/scenelabels.json"
features_dir = "C:/Users/dell/Desktop/good/features"
os.makedirs(features_dir, exist_ok=True)


# %%
cnn = models.resnet50(pretrained=True)
cnn.fc = nn.Identity()
cnn.eval().cuda()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])


# %%
with open(labels_file) as f:
    meta = json.load(f)

folders = {p.name for p in Path(root_dir).iterdir() if p.is_dir()}

entries = []
if isinstance(meta, dict):
    for split in ['train','val','test']:
        if split in meta and isinstance(meta[split], list):
            entries.extend(meta[split])
elif isinstance(meta, list):
    entries = meta
else:
    raise ValueError("scene_labels.json must be dict or list")

shot_label_map = {}
for entry in entries:
    if not isinstance(entry, dict):
        continue
    mid = entry.get('name') or entry.get('movie_id')
    if not mid or mid not in folders:
        continue
    label_list = entry.get('label')
    if not isinstance(label_list, list):
        continue
    shot_label_map[mid] = [(sid, int(lbl)) for sid, lbl in label_list if lbl in ('0','1')]

print(f"Found {len(shot_label_map)} valid movies in folders.")


# %%
import random

movie_ids = list(shot_label_map.keys())
random.seed(42)
random.shuffle(movie_ids)
split = int(0.8 * len(movie_ids))
train_ids = movie_ids[:split]
val_ids = movie_ids[split:]

def count_labels(ids, shot_label_map):
    zeros = ones = 0
    for mid in ids:
        for _, lbl in shot_label_map[mid]:
            zeros += (lbl == 0)
            ones  += (lbl == 1)
    return zeros, ones

train_zeros, train_ones = count_labels(train_ids, shot_label_map)
val_zeros, val_ones = count_labels(val_ids, shot_label_map)

print(f"Training split: {train_zeros} zeros, {train_ones} ones → total={train_zeros+train_ones}")
print(f"Validation split: {val_zeros} zeros, {val_ones} ones → total={val_zeros+val_ones}")

# ✅ remove boost
train_weights = torch.tensor([1.0, 1.5])
val_weights   = torch.tensor([1.0, 1.5])

print(f"Train class weights: {train_weights.tolist()}")
print(f"Val class weights: {val_weights.tolist()}")



# %%
from torch.utils.data import Dataset, DataLoader

class SceneDataset(Dataset):
    def __init__(self, movie_ids, shot_label_map, features_dir, seq_len=10):
        self.movie_ids = movie_ids
        self.shot_label_map = shot_label_map
        self.features_dir = Path(features_dir)
        self.seq_len = seq_len
        self.samples = []
        
        for mid in movie_ids:
            shots = shot_label_map[mid]
            for i in range(len(shots)):
                window = shots[max(0, i - seq_len + 1):i + 1]
                padded = [window[0]] * (self.seq_len - len(window)) + window
                self.samples.append((mid, padded))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mid, window = self.samples[idx]
        feats = []
        labels = []
        mask = []
        
        for sid, lbl in window:
            feat_file = self.features_dir / mid / f"{sid}.pt"
            feat = torch.load(feat_file)
            feats.append(feat)
            labels.append(lbl)
            mask.append(1)
        
        return torch.stack(feats), torch.tensor(labels), torch.tensor(mask)

train_ds = SceneDataset(train_ids, shot_label_map, features_dir)
val_ds = SceneDataset(val_ids, shot_label_map, features_dir)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


# %%
class TransformerSceneBoundary(nn.Module):
    def __init__(self, input_dim=2048, d_model=512, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.dropout_in = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout_out = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, feats, mask):
        x = self.input_proj(feats)
        x = self.bn(x.transpose(1,2)).transpose(1,2)
        x = self.dropout_in(x)
        
        x = self.transformer(x, src_key_padding_mask=~mask.bool())
        x = self.dropout_out(x)
        
        logits = self.classifier(x)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerSceneBoundary().to(device)


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerSceneBoundary().to(device)

# %%
import torch.optim as optim
import torch.nn.functional as F

optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_weights = train_weights.to(device)
val_weights = val_weights.to(device)

best_val_loss = float('inf')

for epoch in range(1, 11):
    model.train()
    criterion_train = nn.CrossEntropyLoss(weight=train_weights.to(device))
    criterion_val = nn.CrossEntropyLoss(weight=val_weights.to(device))
    
    train_loss = 0
    correct = 0
    total = 0
    
    for feats, labels, mask in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(feats, mask)
        
        logits = logits.view(-1, 2)
        labels = labels.view(-1)
        mask_flat = mask.view(-1).bool()
        
        loss = criterion_train(logits[mask_flat], labels[mask_flat])
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = logits.argmax(-1)
        correct += (preds[mask_flat] == labels[mask_flat]).sum().item()
        total += mask_flat.sum().item()
    
    acc = correct / total
    print(f"Epoch {epoch} Train Loss: {train_loss/len(train_loader):.4f} Acc: {acc:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for feats, labels, mask in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
            logits = model(feats, mask)
            logits = logits.view(-1, 2)
            labels = labels.view(-1)
            mask_flat = mask.view(-1).bool()
            
            loss = criterion_val(logits[mask_flat], labels[mask_flat])
            val_loss += loss.item()
            preds = logits.argmax(-1)
            val_correct += (preds[mask_flat] == labels[mask_flat]).sum().item()
            val_total += mask_flat.sum().item()
    
    val_acc = val_correct / val_total
    print(f"Epoch {epoch} Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"best_model22{epoch}.pt")
        print(f"[INFO] Saved best model at epoch {epoch}")


# %%
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re

# ======= CONFIG =======
test_movie_dir = "C:/Users/dell/Desktop/movies/tt0087921"
model_dir = "."
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Load latest best_model =======
model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model221') and f.endswith('.pt')]
if not model_files:
    raise FileNotFoundError("No best_model*.pt found in directory!")
best_model_file = sorted(model_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))[-1]
print(f"[INFO] Loading model: {best_model_file}")

model = TransformerSceneBoundary().to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, best_model_file), map_location=device))
model.eval()

# ======= Load CNN feature extractor =======
cnn = models.resnet50(pretrained=True)
cnn.fc = nn.Identity()
cnn.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

# ======= Get sorted shot IDs =======
shot_ids = sorted(set(p.name.split('_')[1] for p in Path(test_movie_dir).glob("shot_*_img_0.jpg")))
print(f"[INFO] Found {len(shot_ids)} shots in {test_movie_dir}")

# ======= Extract features =======
shot_feats = []
for sid in tqdm(shot_ids, desc="Extracting features"):
    frames = []
    for i in range(3):
        img_path = Path(test_movie_dir) / f"shot_{sid}_img_{i}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            frames.append(transform(img).to(device))
        else:
            print(f"[WARNING] Missing frame: {img_path}")
    if frames:
        frames = torch.stack(frames)
        with torch.no_grad():
            feats = cnn(frames)
        shot_feat = feats.mean(0)
    else:
        shot_feat = torch.zeros(2048).to(device)
    shot_feats.append(shot_feat)

shot_feats = torch.stack(shot_feats).unsqueeze(0)
mask = torch.ones(1, shot_feats.shape[1], dtype=torch.bool, device=device)

# ======= Run inference =======
with torch.no_grad():
    logits = model(shot_feats, mask)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    threshold = 0.75
    preds = (probs[:,1] >= threshold).long().tolist()

print(f"[INFO] Model predicted {sum(preds)} scene boundaries.")

# ======= Build scenes with MIN_SCENE_LEN + MAX_SCENE_LEN =======
MIN_SCENE_LEN = 10
MAX_SCENE_LEN = 50

scenes = []
current_scene = []

for sid, pred in zip(shot_ids, preds):
    current_scene.append(sid)
    if pred == 1 or len(current_scene) >= MAX_SCENE_LEN:
        if len(current_scene) < MIN_SCENE_LEN and scenes:
            scenes[-1].extend(current_scene)  # merge with previous
        else:
            scenes.append(current_scene)
        current_scene = []

if current_scene:
    if len(current_scene) < MIN_SCENE_LEN:
        if scenes:
            scenes[-1].extend(current_scene)
        else:
            scenes.append(current_scene)
    else:
        scenes.append(current_scene)

# Extra: merge first scene forward if too short
if len(scenes) >= 2 and len(scenes[0]) < MIN_SCENE_LEN:
    scenes[1] = scenes[0] + scenes[1]
    scenes = scenes[1:]

print(f"[INFO] Detected {len(scenes)} scenes (after merging short and splitting long scenes).")

# ======= Visualize scenes with 3 frames per shot =======
for idx, scene in enumerate(scenes):
    print(f"\n=== Scene {idx+1} → {len(scene)} shots ===")
    ncols = min(5, len(scene))
    nrows = (len(scene) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, sid in enumerate(scene):
        img_paths = [Path(test_movie_dir) / f"shot_{sid}_img_{j}.jpg" for j in range(3)]
        imgs = []
        for path in img_paths:
            if path.exists():
                imgs.append(np.array(Image.open(path).convert("RGB")))
            else:
                imgs.append(np.ones((224, 224, 3), dtype=np.uint8) * 255)  # white placeholder for missing

        # Concatenate frames horizontally
        combined_img = np.concatenate(imgs, axis=1)
        axes[i].imshow(combined_img)
        axes[i].set_title(f"Shot {sid}")
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(scene), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()



# %%
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re

# ======= CONFIG =======
test_movie_dir = "C:/Users/dell/Downloads/tt0100157/tt0100157"
model_dir = "."
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Load latest best_model =======
model_files = [f for f in os.listdir(model_dir) if f.startswith('best_model221') and f.endswith('.pt')]
if not model_files:
    raise FileNotFoundError("No best_model*.pt found in directory!")
best_model_file = sorted(model_files, key=lambda x: int(re.findall(r'\d+', x)[-1]))[-1]
print(f"[INFO] Loading model: {best_model_file}")

model = TransformerSceneBoundary().to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, best_model_file), map_location=device))
model.eval()

# ======= Load CNN feature extractor =======
cnn = models.resnet50(pretrained=True)
cnn.fc = nn.Identity()
cnn.eval().to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
])

# ======= Get sorted shot IDs =======
shot_ids = sorted(set(p.name.split('_')[1] for p in Path(test_movie_dir).glob("shot_*_img_0.jpg")))
print(f"[INFO] Found {len(shot_ids)} shots in {test_movie_dir}")

# ======= Extract features =======
shot_feats = []
for sid in tqdm(shot_ids, desc="Extracting features"):
    frames = []
    for i in range(3):
        img_path = Path(test_movie_dir) / f"shot_{sid}_img_{i}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            frames.append(transform(img).to(device))
        else:
            print(f"[WARNING] Missing frame: {img_path}")
    if frames:
        frames = torch.stack(frames)
        with torch.no_grad():
            feats = cnn(frames)
        shot_feat = feats.mean(0)
    else:
        shot_feat = torch.zeros(2048).to(device)
    shot_feats.append(shot_feat)

shot_feats = torch.stack(shot_feats).unsqueeze(0)
mask = torch.ones(1, shot_feats.shape[1], dtype=torch.bool, device=device)

# ======= Run inference =======
with torch.no_grad():
    logits = model(shot_feats, mask)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    threshold = 0.75
    preds = (probs[:,1] >= threshold).long().tolist()

print(f"[INFO] Model predicted {sum(preds)} scene boundaries.")

# ======= Build scenes with MIN_SCENE_LEN + MAX_SCENE_LEN =======
MIN_SCENE_LEN = 10
MAX_SCENE_LEN = 50

scenes = []
current_scene = []

for sid, pred in zip(shot_ids, preds):
    current_scene.append(sid)
    if pred == 1 or len(current_scene) >= MAX_SCENE_LEN:
        if len(current_scene) < MIN_SCENE_LEN and scenes:
            scenes[-1].extend(current_scene)  # merge with previous
        else:
            scenes.append(current_scene)
        current_scene = []

if current_scene:
    if len(current_scene) < MIN_SCENE_LEN:
        if scenes:
            scenes[-1].extend(current_scene)
        else:
            scenes.append(current_scene)
    else:
        scenes.append(current_scene)

# Extra: merge first scene forward if too short
if len(scenes) >= 2 and len(scenes[0]) < MIN_SCENE_LEN:
    scenes[1] = scenes[0] + scenes[1]
    scenes = scenes[1:]

print(f"[INFO] Detected {len(scenes)} scenes (after merging short and splitting long scenes).")




