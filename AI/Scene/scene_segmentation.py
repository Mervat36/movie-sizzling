import os
import sys
import json
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Shot Encoder (uses pretrained ResNet50)
class ShotEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(base_model.children())[:-1]
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        feats = self.features(x).view(B, F, -1)
        return feats

# CRN model (Transformer over 3-frame features)
class CRNModel(nn.Module):
    def __init__(self, dim=2048, heads=4, layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(dim, 1)
        self.encoder = ShotEncoder()

    def forward(self, x):
        x = self.encoder(x)  # [B, 3, 2048]
        x = self.transformer(x).mean(dim=1)  # [B, 2048]
        return self.classifier(x).squeeze(1)

# Prediction pipeline
def predict_scene_segments(shot_folder, model_path, output_path, threshold=0.7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNModel().to(device)

    # Adjusted loading with key name fix
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("crn.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()


    filenames = os.listdir(shot_folder)
    shot_ids = sorted(set("_".join(name.split("_")[:2]) for name in filenames if name.endswith(".jpg")))

    predictions = []

    for shot_id in shot_ids:
        paths = [
            os.path.join(shot_folder, f"{shot_id}_start.jpg"),
            os.path.join(shot_folder, f"{shot_id}_middle.jpg"),
            os.path.join(shot_folder, f"{shot_id}_end.jpg")
        ]

        frames = []
        for path in paths:
            if not os.path.exists(path):
                print(f"Missing frame: {path}")
                break
            image = Image.open(path).convert("RGB")
            image = transform(image)
            frames.append(image)

        if len(frames) != 3:
            continue

        image_tensor = torch.stack(frames).unsqueeze(0).to(device)  # [1, 3, C, H, W]
        with torch.no_grad():
            output = model(image_tensor)
            score = torch.sigmoid(output).item()
            predictions.append((shot_id, score))

    # Group into scenes
    scenes, current_scene = [], []
    for shot_id, score in predictions:
        current_scene.append(shot_id)
        if score > threshold:
            scenes.append(current_scene)
            current_scene = []
    if current_scene:
        scenes.append(current_scene)

    with open(output_path, "w") as f:
        json.dump({"scenes": scenes}, f, indent=2)

    print(f"✅ Scene segmentation complete. Results saved to {output_path}")

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scene_segmentation.py <shot_folder> <model_path> <output_path>")
        sys.exit(1)

    shot_folder = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    predict_scene_segments(shot_folder, model_path, output_path)
