# === SceneSegmentation.py ===
import os, json, torch, torchvision
from PIL import Image
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# === Configuration ===
DATA_ROOT = "D:/Dataset/movies"
SCENE_LABELS_PATH = "D:/Dataset/scenelabels_final.json"
MODEL_SAVE_DIR = "./checkpoints"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5         # L2 regularization
SHOT_NUM = 8
SIM_CHANNEL = 128
LSTM_HIDDEN_SIZE = 128
BIDIRECTIONAL = True
PIN_MEMORY = True
MAX_MOVIES = 95
FORCE_EARLY_SAMPLES = 10
CONFIDENCE_THRESHOLD = 0.85

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Augmentation ===
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# === Dataset ===
class LGSSDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.transform = transform or data_transforms
        self.samples = []
        for movie in labels[:MAX_MOVIES]:
            name = movie["name"]
            shots = sorted(movie["label"], key=lambda x: int(x[0]))
            valid = []
            for sid_str, lbl in shots:
                if lbl == "-1": continue
                sid = int(sid_str)
                shot_dir = os.path.join(DATA_ROOT, name, name)
                paths = [os.path.join(shot_dir, f"shot_{sid:04d}_img_{i}.jpg") for i in (0,1,2)]
                if all(os.path.exists(p) for p in paths):
                    valid.append((paths, int(lbl)))
            R = len(valid) - SHOT_NUM
            if R <= 0: continue
            print(f"🎬 {name}: using {R} training samples")
            for i in range(R):
                group = valid[i:i+SHOT_NUM]
                lbls = [g[1] for g in group]
                bin_lbl = [int(lbls[k]!=lbls[k+1]) for k in range(SHOT_NUM-1)]
                self.samples.append(([g[0] for g in group], bin_lbl[SHOT_NUM//2-1]))
            for i in range(min(FORCE_EARLY_SAMPLES, R)):
                group = valid[i:i+SHOT_NUM]
                lbls = [g[1] for g in group]
                bin_lbl = [int(lbls[k]!=lbls[k+1]) for k in range(SHOT_NUM-1)]
                self.samples.append(([g[0] for g in group], bin_lbl[SHOT_NUM//2-1]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        shots = []
        for pset in paths:
            frames = []
            for p in pset:
                try:
                    # Add error handling for file loading
                    if not os.path.exists(p):
                        print(f"⚠️  Warning: File not found: {p}")
                        continue
                    img = Image.open(p).convert("RGB")
                    frames.append(self.transform(img))
                except Exception as e:
                    print(f"❌ Error loading image {p}: {e}")
                    continue
            
            # Only proceed if we have frames
            if frames:
                avg = torch.stack(frames).mean(0)
                shots.append(avg)
            else:
                # If no frames loaded, create a zero tensor as fallback
                print(f"⚠️  No frames loaded for paths: {pset}")
                # Create a dummy tensor with correct shape (3, 224, 224)
                dummy_frame = torch.zeros(3, 224, 224)
                shots.append(dummy_frame)
        
        return torch.stack(shots), torch.tensor(label, dtype=torch.long)

# === Model with Dropout ===
class ShotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.dim = resnet.fc.in_features
    def forward(self, x):
        B,S,C,H,W = x.shape
        x = x.view(B*S, C, H, W)
        f = self.features(x).view(B, S, -1)
        return f

class CosSimBlock(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.lin = nn.Linear((SHOT_NUM//2)*dim, SIM_CHANNEL)
    def forward(self,x):
        B,S,D = x.shape
        p1,p2 = torch.split(x, [SHOT_NUM//2]*2, dim=1)
        v1 = self.lin(p1.reshape(B,-1))
        v2 = self.lin(p2.reshape(B,-1))
        return F.cosine_similarity(v1,v2,dim=1)

class LGSSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = ShotEncoder()
        self.cs  = CosSimBlock(self.enc.dim)
        self.lstm= nn.LSTM(
            input_size=self.enc.dim+SIM_CHANNEL,
            hidden_size=LSTM_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=BIDIRECTIONAL
        )
        out_dim = LSTM_HIDDEN_SIZE*(2 if BIDIRECTIONAL else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim,100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100,2)
        )
    def forward(self,x):
        f = self.enc(x)                                          # (B,S,D)
        sim = self.cs(f).unsqueeze(1).unsqueeze(2)               # (B,1,1)
        sim = sim.expand(-1, SHOT_NUM-1, SIM_CHANNEL)           # (B,S-1,CS)
        cat = torch.cat([f[:,:SHOT_NUM-1,:], sim], dim=2)       # (B,S-1,D+CS)
        out,_ = self.lstm(cat)
        return self.fc(out[:,SHOT_NUM//2-1,:])

# === Training with Early Stopping & LR Scheduler ===
def train():
    with open(SCENE_LABELS_PATH) as f:
        data = json.load(f)["train"]

    # tell me how many movies we're actually using
    n_movies = min(MAX_MOVIES, len(data))
    print(f"🔢 Training on {n_movies} movies (MAX_MOVIES={MAX_MOVIES}, total in file={len(data)})")

    # split into train / val
    tr, vl = train_test_split(data, test_size=0.2, random_state=42)
    train_ds = LGSSDataset(tr)
    val_ds   = LGSSDataset(vl)

    tr_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    vl_ld = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = LGSSModel().to(device)
    opt   = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)
    sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
    crit  = nn.CrossEntropyLoss(weight=torch.tensor([0.5,2.0]).to(device))

    best_val = float('inf')
    wait = 0
    for ep in range(1, NUM_EPOCHS+1):
        # — Train —
        model.train()
        t_loss=0; correct=0
        for x,y in tqdm(tr_ld, desc=f"Epoch {ep} Train"):
            x,y = x.to(device), y.to(device)
            p    = model(x)
            loss = crit(p,y)
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss+= loss.item()
            correct += (p.argmax(1)==y).sum().item()
        train_acc = correct/len(train_ds)
        print(f"Train  L={t_loss/len(tr_ld):.4f} Acc={train_acc:.4f}")

        # — Validate —
        model.eval()
        v_loss=0; v_corr=0
        with torch.no_grad():
            for x,y in tqdm(vl_ld, desc=f"Epoch {ep} Val"):
                x,y = x.to(device), y.to(device)
                p    = model(x)
                v_loss+= crit(p,y).item()
                v_corr+= (p.argmax(1)==y).sum().item()
        val_loss = v_loss/len(vl_ld)
        val_acc  = v_corr/len(val_ds)
        print(f"Val    L={val_loss:.4f} Acc={val_acc:.4f}")

        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            wait=0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_SAVE_DIR,"best_model.pt"))
            print("✅ Saved new best")
        else:
            wait+=1
            if wait>=3:
                print("⏹ Early stopping")
                break

if __name__=='__main__':
    train()