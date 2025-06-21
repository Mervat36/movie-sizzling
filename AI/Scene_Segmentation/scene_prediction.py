#!/usr/bin/env python3
"""
Smart Scene Detection - Automatically finds optimal parameters for each movie
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import shutil

# === Constants ===
SHOT_NUM = 8
CONFIDENCE_THRESHOLD = 0.2
MIN_BOUNDARY_GAP = 2
BATCH_SIZE = 1
MODEL_PATH = "./best_model.pt"
MOVIE_FOLDER = "heross22"
OUTPUT_SCENES_JSON = "scenes.json"
SCENE_SHEETS_DIR = "scene_sheets"
BIDIRECTIONAL = True
LSTM_HIDDEN_SIZE = 128
SIM_CHANNEL = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        print(f"[STATS] Using {len(self.shot_ids)} complete shots (out of {total_shots} total)")
        
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
    print(f"[BOUNDARIES] Boundaries (shot IDs): {bounds}")
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
        if not scene:
            continue
            
        scene_id = f"scene_{idx:03d}"
        
        # Save individual scene thumbnail (first shot of the scene)
        first_shot = scene[0]
        start_frame_path = os.path.join(folder, f"shot_{first_shot:03d}_start.jpg")
        if os.path.exists(start_frame_path):
            try:
                img = Image.open(start_frame_path).convert("RGB").resize(thumb_size)
                thumb_path = os.path.join(output_dir, f"{scene_id}.jpg")
                img.save(thumb_path, "JPEG", quality=95)
                print(f"[SAVED] Saved scene {idx:03d} thumbnail: {thumb_path}")
            except Exception as e:
                print(f"[ERROR] Error saving scene thumbnail {thumb_path}: {e}")
                # Create a dummy thumbnail
                dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
                dummy_img.save(thumb_path, "JPEG")
        else:
            print(f"[WARNING] Start frame not found for scene {idx}: {start_frame_path}")
            # Create a dummy thumbnail
            dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
            thumb_path = os.path.join(output_dir, f"{scene_id}.jpg")
            dummy_img.save(thumb_path, "JPEG")
        
        # Create individual shot images for each scene (for upload controller compatibility)
        scene_shot_dir = os.path.join(output_dir, scene_id)
        os.makedirs(scene_shot_dir, exist_ok=True)
        
        for shot_num in scene:
            for view in ["start", "middle", "end"]:
                source_path = os.path.join(folder, f"shot_{shot_num:03d}_{view}.jpg")
                if os.path.exists(source_path):
                    try:
                        # Copy the shot image to scene directory with new naming
                        target_filename = f"{scene_id}_shot{shot_num:03d}_{view}.jpg"
                        target_path = os.path.join(scene_shot_dir, target_filename)
                        
                        # Copy the file
                        shutil.copy2(source_path, target_path)
                        print(f"[SAVED] {target_filename}")
                    except Exception as e:
                        print(f"[ERROR] Error copying {source_path}: {e}")
                else:
                    print(f"[WARNING] Shot file not found: {source_path}")
        
        # Also save scene sheet for reference (optional)
        thumbs = []
        for sid in scene:
            # Updated to use the actual file format: shot_XXX_start.jpg
            path = os.path.join(folder, f"shot_{sid:03d}_start.jpg")
            try:
                if os.path.exists(path):
                    img = Image.open(path).convert("RGB").resize(thumb_size)
                    thumbs.append(img)
                else:
                    print(f"[WARNING] Warning: File not found for scene sheet: {path}")
                    # Create a dummy thumbnail
                    dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
                    thumbs.append(dummy_img)
            except Exception as e:
                print(f"[ERROR] Error loading image for scene sheet {path}: {e}")
                # Create a dummy thumbnail
                dummy_img = Image.new("RGB", thumb_size, (128, 128, 128))
                thumbs.append(dummy_img)
        
        # compose horizontal sheet (optional, for debugging)
        if thumbs:
            w, h = thumb_size
            sheet = Image.new("RGB", (w*len(thumbs), h))
            for i, thumb in enumerate(thumbs):
                sheet.paste(thumb, (i*w, 0))
            sheet_path = os.path.join(output_dir, f"{scene_id}_sheet.png")
            sheet.save(sheet_path)
            print(f"[SAVED] Saved scene {idx:03d} sheet ({len(thumbs)} shots): {sheet_path}")


# === Main ===
# (Main execution moved to end of file)

def get_confidence_scores(movie_folder, model_path):
    """Get confidence scores for all shots in a movie"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset and model
    dataset = MovieSceneTestDataset(movie_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    shot_ids = dataset.shot_ids
    
    model = LGSSModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Get all confidence scores
    confidences = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            x = batch.to(device)
            logits = model(x)
            prob_boundary = F.softmax(logits, dim=1)[0,1].item()
            
            center_idx = idx + 4  # SHOT_NUM//2 - 1
            if center_idx < len(shot_ids):
                sid = shot_ids[center_idx]
                confidences.append((sid, prob_boundary))
    
    return confidences, shot_ids

def detect_boundaries(confidences, threshold, min_gap):
    """Detect boundaries with given parameters"""
    # Filter by threshold
    candidates = [(sid, conf) for sid, conf in confidences if conf >= threshold]
    
    # Peak detection
    peaks = []
    for i, (sid, conf) in enumerate(candidates):
        is_peak = True
        
        # Check previous candidates
        for j in range(i-1, max(0, i-3), -1):
            if j < len(candidates) and candidates[j][1] >= conf:
                is_peak = False
                break
        
        # Check next candidates
        for j in range(i+1, min(len(candidates), i+3)):
            if j < len(candidates) and candidates[j][1] >= conf:
                is_peak = False
                break
        
        if is_peak:
            peaks.append(sid)
    
    # Apply minimum gap
    final_boundaries = []
    for peak in sorted(peaks):
        if not final_boundaries or (peak - final_boundaries[-1]) >= min_gap:
            final_boundaries.append(peak)
    
    return final_boundaries

def calculate_scene_quality(scenes):
    """Calculate quality metrics for scene segmentation"""
    if not scenes:
        return 0, 0, 0
    
    scene_lengths = [len(scene) for scene in scenes]
    num_scenes = len(scenes)
    total_shots = sum(scene_lengths)
    
    # Quality metrics
    avg_length = np.mean(scene_lengths)
    std_length = np.std(scene_lengths)
    min_length = min(scene_lengths)
    max_length = max(scene_lengths)
    
    # Base quality score
    quality_score = 100
    
    # Penalize very short scenes (less than 2 shots)
    short_penalty = sum(1 for length in scene_lengths if length < 2) * 15
    
    # Penalize very long scenes (more than 40 shots) - but less harshly
    long_penalty = sum(1 for length in scene_lengths if length > 40) * 5
    
    # Balance score (lower is better)
    balance_score = std_length / avg_length if avg_length > 0 else float('inf')
    balance_penalty = balance_score * 15
    
    # Target scene count based on movie length
    if total_shots <= 20:
        target_scenes = 1
    elif total_shots <= 40:
        target_scenes = 2
    elif total_shots <= 60:
        target_scenes = 3
    elif total_shots <= 80:
        target_scenes = 4
    else:
        target_scenes = 5
    
    # Scene count penalty/bonus
    scene_count_diff = abs(num_scenes - target_scenes)
    scene_count_penalty = scene_count_diff * 10
    
    # Prefer multiple scenes over single scene for longer movies
    scene_count_bonus = 0
    if num_scenes == 1 and total_shots > 20:
        scene_count_bonus = -20
    elif num_scenes >= 2 and num_scenes <= 6:
        scene_count_bonus = 10
    
    # Calculate final quality
    quality_score = quality_score - short_penalty - long_penalty - balance_penalty - scene_count_penalty + scene_count_bonus
    quality_score = max(0.0, float(quality_score))
    
    return quality_score, avg_length, std_length

def find_optimal_parameters(movie_folder, model_path):
    """Find optimal parameters for a specific movie"""
    print(f"[MOVIE] Smart Scene Detection for: {movie_folder}")
    print("=" * 60)
    
    # Get confidence scores
    confidences, shot_ids = get_confidence_scores(movie_folder, model_path)
    scores = [conf for _, conf in confidences]
    
    print(f"[STATS] Analyzing {len(shot_ids)} shots")
    print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"   Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    # Define parameter ranges to test
    thresholds = np.arange(0.05, 0.8, 0.05)  # 0.05 to 0.75 in steps of 0.05 (lower start)
    gaps = [1, 2, 3, 4, 5]
    
    best_config = None
    best_quality = -1
    results = []
    
    print(f"\n[TEST] Testing {len(thresholds)} thresholds x {len(gaps)} gaps = {len(thresholds) * len(gaps)} configurations...")
    
    for threshold in thresholds:
        for gap in gaps:
            # Detect boundaries
            boundaries = detect_boundaries(confidences, threshold, gap)
            
            # Group into scenes
            scenes = group_scenes(shot_ids, boundaries)
            
            # Calculate quality
            quality, avg_length, std_length = calculate_scene_quality(scenes)
            
            # Store result
            result = {
                "threshold": threshold,
                "gap": gap,
                "boundaries": boundaries,
                "num_scenes": len(scenes),
                "quality": quality,
                "avg_length": avg_length,
                "std_length": std_length,
                "scenes": scenes
            }
            results.append(result)
            
            # Update best if better
            if quality > best_quality:
                best_quality = quality
                best_config = result
    
    # Sort results by quality
    results.sort(key=lambda x: x["quality"], reverse=True)
    
    # Show top 5 results
    print(f"\n[TOP] Top 5 configurations:")
    for i, result in enumerate(results[:5]):
        print(f"   {i+1}. Threshold: {result['threshold']:.2f}, Gap: {result['gap']}")
        print(f"      Quality: {result['quality']:.1f}, Scenes: {result['num_scenes']}")
        print(f"      Avg length: {result['avg_length']:.1f} ± {result['std_length']:.1f}")
        print(f"      Boundaries: {result['boundaries']}")
    
    return best_config, results

def apply_smart_detection(movie_folder, model_path, save_results=True):
    """Apply smart scene detection to a movie"""
    # Find optimal parameters
    best_config, all_results = find_optimal_parameters(movie_folder, model_path)
    
    if not best_config:
        print("[ERROR] No valid configuration found!")
        return None
    
    print(f"\n[SUCCESS] Best configuration selected:")
    print(f"   Threshold: {best_config['threshold']:.3f}")
    print(f"   Min gap: {best_config['gap']}")
    print(f"   Quality score: {best_config['quality']:.1f}")
    print(f"   Number of scenes: {best_config['num_scenes']}")
    
    # Show scene breakdown
    print(f"\n[SCENES] Scene breakdown:")
    for i, scene in enumerate(best_config['scenes'], 1):
        if scene:
            print(f"   Scene {i}: shots {scene[0]} to {scene[-1]} ({len(scene)} shots)")
    
    # Save scenes in the format expected by upload controller
    movie_title = os.path.basename(movie_folder)
    output_dir = os.path.join("output", movie_title)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read shot data to get timestamps
    shot_json_path = os.path.join("output", f"{movie_title}_shots.json")
    timestamps = {}
    if os.path.exists(shot_json_path):
        try:
            with open(shot_json_path, "r") as f:
                shot_data = json.load(f)
            
            for shot in shot_data.get("shots", []):
                shot_id = str(shot.get("shotNumber", ""))
                timestamps[shot_id] = {
                    "start_time": shot.get("startTime", ""),
                    "end_time": shot.get("endTime", "")
                }
            print(f"[STATS] Loaded timestamps for {len(timestamps)} shots")
        except Exception as e:
            print(f"[WARNING] Warning: Could not load shot timestamps: {e}")
    
    # Create scenes.json in the expected format
    scenes_data = []
    for idx, scene in enumerate(best_config['scenes'], 1):
        if scene:
            scene_id = f"scene_{idx:03d}"
            start_shot = str(scene[0])
            end_shot = str(scene[-1])
            
            # Get timestamps from shot data
            start_time = timestamps.get(start_shot, {}).get("start_time", "")
            end_time = timestamps.get(end_shot, {}).get("end_time", "")
            
            # Create thumbnail path
            thumbnail_path = f"scene-results/{movie_title}/{scene_id}.jpg"
            
            scene_info = {
                "scene_id": scene_id,
                "start_shot": start_shot,
                "end_shot": end_shot,
                "start_time": start_time,
                "end_time": end_time,
                "thumbnail_path": thumbnail_path
            }
            scenes_data.append(scene_info)
    
    # Save scenes.json
    scenes_json_path = os.path.join(output_dir, "scenes.json")
    with open(scenes_json_path, "w") as f:
        json.dump({"title": movie_title, "scenes": scenes_data}, f, indent=2)
    
    print(f"[SAVED] Scenes saved to: {scenes_json_path}")
    
    # Save scene thumbnails
    scene_sheets_dir = os.path.join(output_dir, "scenes")
    save_scene_sheets(best_config['scenes'], movie_folder, scene_sheets_dir)
    
    # Save results if requested
    if save_results:
        output_data = {
            "movie_folder": movie_folder,
            "optimal_config": best_config,
            "all_results": all_results[:10],  # Top 10 results
            "timestamp": str(np.datetime64('now'))
        }
        
        with open("smart_detection_results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[SAVED] Results saved to: smart_detection_results.json")
    
    return best_config

def batch_process_movies(movie_folders, model_path):
    """Process multiple movies with smart detection"""
    print(f"[BATCH] Batch Processing {len(movie_folders)} movies")
    print("=" * 60)
    
    results = {}
    
    for folder in movie_folders:
        if os.path.exists(folder):
            print(f"\n[FOLDER] Processing: {folder}")
            try:
                result = apply_smart_detection(folder, model_path, save_results=False)
                results[folder] = result
            except Exception as e:
                print(f"[ERROR] Error processing {folder}: {e}")
                results[folder] = None
        else:
            print(f"[ERROR] Folder not found: {folder}")
            results[folder] = None
    
    # Save batch results
    with open("batch_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SAVED] Batch results saved to: batch_detection_results.json")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Scene Detection')
    parser.add_argument('--movie_folder', type=str, default="heross22",
                        help='Movie folder to process')
    parser.add_argument('--model_path', type=str, default="./best_model.pt",
                        help='Path to model file')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple movies')
    
    args = parser.parse_args()
    
    if args.batch:
        # Process multiple movies
        movie_folders = ["heross22", "The_Vampire_Diaries"]  # Add more as needed
        batch_process_movies(movie_folders, args.model_path)
    else:
        # Process single movie
        apply_smart_detection(args.movie_folder, args.model_path) 