# import os
# import sys
# import torch
# import numpy as np
# import json
# from PIL import Image
# from omegaconf import OmegaConf
# from model.shot_encoder.resnet import resnet50
# from model.crn.trn import TransformerCRN
# from transform.to_tensor import VideoToTensor as ToTensor
# from transform.random_resized_crop import VideoRandomResizedCrop as ResizedCenterCrop
# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import cosine_similarity

# # ========================= Model Loading =========================
# def load_model(cfg_path, checkpoint_path):
#     cfg = OmegaConf.load(cfg_path)
#     shot_encoder = resnet50(pretrained=False)
#     crn = TransformerCRN(cfg.MODEL.contextual_relation_network.params["trn"])

#     checkpoint = torch.load(checkpoint_path, map_location="cpu")
#     state_dict = checkpoint.get("state_dict", {})

#     shot_encoder.load_state_dict(
#         {k.replace("shot_encoder.", ""): v for k, v in state_dict.items() if "shot_encoder" in k}, strict=False)
#     crn.load_state_dict(
#         {k.replace("crn.", ""): v for k, v in state_dict.items() if "crn" in k}, strict=False)

#     shot_encoder.eval()
#     crn.eval()
#     return shot_encoder, crn

# # ========================= Frame Preprocessing =========================
# def preprocess_frames(images):
#     resize_crop = ResizedCenterCrop(224)
#     to_tensor = ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     frame_list = resize_crop(images)
#     frames_tensor = torch.stack([to_tensor([frame])[0] for frame in frame_list])
#     return frames_tensor.mean(dim=0)

# # ========================= Inference =========================
# def run_inference(shot_encoder, crn, shots):
#     all_scene_features = []
#     rep_images = []

#     for shot_id, images in shots.items():
#         input_tensor = preprocess_frames(images).unsqueeze(0)
#         rep_images.append(images[0])

#         with torch.no_grad():
#             shot_features = shot_encoder(input_tensor)
#             shot_features = shot_features.unsqueeze(1)
#             scene_features, _ = crn(shot_features)
#             all_scene_features.append(scene_features.squeeze(0).mean(dim=0).numpy())

#     return np.array(all_scene_features), rep_images

# # ========================= Scene Segmentation =========================
# def segment_scenes(features, threshold=0.85, k=5, temporal_window=15):
#     scenes = []
#     num_shots = len(features)
#     used = set()
#     i = 0
#     while i < num_shots:
#         start = i
#         start_idx = max(0, i - temporal_window)
#         end_idx = min(num_shots, i + temporal_window + 1)
#         knn = NearestNeighbors(n_neighbors=min(k, end_idx - start_idx), metric='euclidean')
#         knn.fit(features[start_idx:end_idx])
#         indices = knn.kneighbors([features[i]], return_distance=False)[0] + start_idx
#         end = max(indices)
#         scenes.append((start, min(end, num_shots - 1)))
#         used.update(range(start, end + 1))
#         i = end + 1
#     return scenes

# # ========================= Main Entry Point =========================
# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage: python scene_segmentation.py <title> <json_path>")
#         sys.exit(1)

#     title = sys.argv[1]
#     json_path = sys.argv[2]

#     CONFIG_PATH = "AI/Scene/bassl/bassl/checkpoints/config.json"
#     CHECKPOINT_PATH = "AI/Scene/bassl/bassl/checkpoints/model-v1.ckpt"
#     OUTPUT_DIR = os.path.join("output", title, "scenes")
#     SHOT_DIR = os.path.join("shots", title)
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     with open(json_path, "r") as f:
#         shot_data = json.load(f)

#     shots = {}
#     shot_id_list = []
#     for shot in shot_data["shots"]:
#         shot_id = shot["shotNumber"]
#         images = []
#         for img_path in shot["images"].values():
#             full_path = os.path.join(SHOT_DIR, os.path.basename(img_path))
#             img = Image.open(full_path).convert("RGB")
#             images.append(img)
#         shots[shot_id] = images
#         shot_id_list.append(shot_id)

#     shot_encoder, crn = load_model(CONFIG_PATH, CHECKPOINT_PATH)
#     scene_features, rep_images = run_inference(shot_encoder, crn, shots)
#     scenes = segment_scenes(scene_features)

#     results = []
#     for idx, (start, end) in enumerate(scenes):
#         scene_id = f"scene_{idx+1:03}"
#         start_shot = shot_id_list[start]
#         end_shot = shot_id_list[end]

#         thumb_path = os.path.join(OUTPUT_DIR, f"{scene_id}.jpg")
#         rep_images[start].save(thumb_path)

#         results.append({
#             "scene_id": scene_id,
#             "start_shot": start_shot,
#             "end_shot": end_shot,
#             "start_time": shot_data["shots"][start].get("startTime", ""),
#             "end_time": shot_data["shots"][end].get("endTime", ""),
#             "thumbnail_path": f"scene-results/{title}/{scene_id}.jpg"
#         })

#     with open(os.path.join("output", title, "scenes.json"), "w") as f:
#         json.dump({"title": title, "scenes": results}, f, indent=2)

#     print("Scene segmentation completed. Scenes saved to output.")

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from model.shot_encoder.resnet import resnet50
from model.crn.trn import TransformerCRN
from transform.to_tensor import VideoToTensor as ToTensor
from transform.random_resized_crop import VideoRandomResizedCrop as ResizedCenterCrop
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import shutil


def load_model(cfg_path, checkpoint_path):
    print("Loading configuration and model checkpoint...")
    cfg = OmegaConf.load(cfg_path)
    shot_encoder = resnet50(pretrained=False)
    crn = TransformerCRN(cfg.MODEL.contextual_relation_network.params["trn"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})

    shot_encoder.load_state_dict({k.replace("shot_encoder.", ""): v for k, v in state_dict.items() if "shot_encoder" in k}, strict=False)
    crn.load_state_dict({k.replace("crn.", ""): v for k, v in state_dict.items() if "crn" in k}, strict=False)

    shot_encoder.eval()
    crn.eval()
    print("Model loaded successfully.")
    return shot_encoder, crn


def preprocess_frames(image_list):
    print(f"Processing {len(image_list)} frames...")
    resize_crop = ResizedCenterCrop(224)
    to_tensor = ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    frames = []
    for frame in image_list:
        if not isinstance(frame, Image.Image):
            print("Warning: Invalid image type encountered. Skipping.")
            continue
        frame_list = resize_crop([frame])
        frame_tensor = to_tensor(frame_list)[0]
        frames.append(frame_tensor)

    if not frames:
        raise ValueError("No valid frames found for preprocessing.")

    frames_tensor = torch.stack(frames)
    averaged_frame = frames_tensor.mean(dim=0)
    print(f"Preprocessing completed for {len(frames)} frames.")
    return averaged_frame


def run_inference(shot_encoder, crn, shots):
    print("\nRunning inference on shots...")
    all_scene_features = []
    rep_images = []
    for shot_id in sorted(shots.keys(), key=lambda x: int(x)):
        image_paths = [img for img, _ in shots[shot_id]]
        print(f"\nProcessing Shot {shot_id} - {len(image_paths)} frames")
        input_tensor = preprocess_frames(image_paths).unsqueeze(0)
        rep_img = image_paths[0]
        rep_images.append(rep_img)

        with torch.no_grad():
            shot_features = shot_encoder(input_tensor)
            shot_features = shot_features.unsqueeze(1)
            print(f"Shot {shot_id} - Input Shape to CRN: {shot_features.shape}")
            scene_features, _ = crn(shot_features)
            print(f"Shot {shot_id} - Output Shape from CRN: {scene_features.shape}")
            all_scene_features.append(scene_features.squeeze(0).mean(dim=0).numpy())

    print("\nInference completed.")
    return np.array(all_scene_features), rep_images


def segment_scenes(features, threshold=0.85, k=5, temporal_window=15):
    print(f"\nSegmenting scenes using threshold={threshold:.4f}...")
    scenes = []
    num_shots = len(features)
    used_shots = set()
    i = 0
    while i < num_shots:
        start_shot = i
        start_idx = max(0, i - temporal_window)
        end_idx = min(num_shots, i + temporal_window + 1)
        temporal_features = features[start_idx:end_idx]

        actual_k = min(k, len(temporal_features))
        if actual_k <= 1:
            break

        knn = NearestNeighbors(n_neighbors=actual_k, metric='euclidean')
        knn.fit(temporal_features)
        distances, indices = knn.kneighbors([features[i]], return_distance=True)
        if len(indices) == 0:
            break

        relative_indices = sorted(indices[0] + start_idx)
        end_shot = max(relative_indices)
        while end_shot in used_shots and end_shot + 1 < num_shots:
            end_shot += 1

        if end_shot < num_shots - 1:
            similarity = cosine_similarity([features[end_shot]], [features[end_shot + 1]])[0][0]
            if similarity > threshold:
                end_shot += 1

        scenes.append((start_shot, min(end_shot, num_shots - 1)))
        used_shots.update(range(start_shot, end_shot + 1))
        print(f"Scene {len(scenes)}: Shot {start_shot} to Shot {end_shot}")
        i = end_shot + 1

    print("\nScene segmentation completed.")
    return scenes


if __name__ == "__main__":
    print("scene_segmentation.py started.")
    print("Received arguments:", sys.argv)

    if len(sys.argv) < 4:
        print("Not enough arguments. Expected: <title> <json_path> <shot_folder>")
        sys.exit(1)

    title = sys.argv[1]
    json_path = sys.argv[2]
    SHOT_DIR = sys.argv[3]

    CONFIG_PATH = "AI/Scene/model/checkpoints/config.json"
    CHECKPOINT_PATH = "AI/Scene/model/checkpoints/model-v1.ckpt"
    OUTPUT_DIR = os.path.join("output", title, "scenes")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(json_path):
        print(f"JSON file not found at path: {json_path}")
        sys.exit(1)

    if not os.path.exists(SHOT_DIR):
        print(f"Shot folder not found at path: {SHOT_DIR}")
        sys.exit(1)

    with open(json_path, "r") as f:
        shot_data = json.load(f)

    timestamps = {}
    for shot in shot_data["shots"]:
        shot_id = str(shot["shotNumber"]).lstrip("0") or "0"
        timestamps[shot_id] = {
            "start_time": shot.get("startTime", ""),
            "end_time": shot.get("endTime", "")
        }

    shots = defaultdict(list)
    pattern = re.compile(r"(shot_(\d+)_(start|middle|end)\.jpg)")
    for filename in sorted(os.listdir(SHOT_DIR)):
        match = pattern.match(filename)
        if match:
            full_name = match.group(1)
            shot_id = match.group(2).lstrip("0") or "0"
            pos = match.group(3)
            path = os.path.join(SHOT_DIR, filename)
            img = Image.open(path).convert("RGB")
            shots[shot_id].append((img, pos))

    valid_shot_ids = sorted([sid for sid in shots if sid in timestamps], key=lambda x: int(x))
    if not valid_shot_ids:
        print("No valid shots with timestamps found. Exiting.")
        sys.exit(1)

    print(f"Valid shots loaded: {len(valid_shot_ids)}")

    shot_encoder, crn = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    scene_features, rep_images = run_inference(shot_encoder, crn, {sid: shots[sid] for sid in valid_shot_ids})
    scenes = segment_scenes(scene_features)

    results = []
    for scene_index, (start, end) in enumerate(scenes):
        scene_id = f"scene_{scene_index+1:03}"
        start_shot = valid_shot_ids[start]
        end_shot = valid_shot_ids[end]

        # Save thumbnail
        thumb_path = os.path.join(OUTPUT_DIR, f"{scene_id}.jpg")
        rep_images[start].save(thumb_path)

        # Save shots under scene folder with clear filenames
        scene_folder = os.path.join(OUTPUT_DIR, scene_id)
        os.makedirs(scene_folder, exist_ok=True)
        for sid in valid_shot_ids[start:end + 1]:
            for img, pos in shots[sid]:
                final_name = f"{scene_id}_shot{int(sid):03}_{pos}.jpg"
                img.save(os.path.join(scene_folder, final_name))

        results.append({
            "scene_id": scene_id,
            "start_shot": start_shot,
            "end_shot": end_shot,
            "start_time": timestamps[start_shot]["start_time"],
            "end_time": timestamps[end_shot]["end_time"],
            "thumbnail_path": f"scene-results/{title}/{scene_id}.jpg"
        })

    with open(os.path.join("output", title, "scenes.json"), "w") as f:
        json.dump({"title": title, "scenes": results}, f, indent=2)

    print("Scene segmentation completed.")
