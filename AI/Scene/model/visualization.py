import os
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


def load_model(cfg_path, checkpoint_path):
    """ Load the Shot Encoder and CRN """
    print("🔍 Loading configuration and model checkpoint...")
    cfg = OmegaConf.load(cfg_path)

    shot_encoder = resnet50(pretrained=False)
    crn = TransformerCRN(cfg.MODEL.contextual_relation_network.params["trn"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})

    shot_encoder.load_state_dict(
        {k.replace("shot_encoder.", ""): v for k, v in state_dict.items() if "shot_encoder" in k}, strict=False)
    crn.load_state_dict({k.replace("crn.", ""): v for k, v in state_dict.items() if "crn" in k}, strict=False)

    shot_encoder.eval()
    crn.eval()
    print("✅ Model loaded successfully.")
    return shot_encoder, crn


def preprocess_frames(image_paths):
    """ Load and preprocess key frames for a shot """
    print(f"📸 Processing {len(image_paths)} frames...")
    resize_crop = ResizedCenterCrop(224)
    to_tensor = ToTensor(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    frames = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found - {img_path}")
            continue
        frame = Image.open(img_path).convert("RGB")
        frame_list = resize_crop([frame])
        frame_tensor = to_tensor(frame_list)[0]
        frames.append(frame_tensor)

    if len(frames) == 0:
        raise ValueError("❌ No valid frames found for preprocessing.")

    frames_tensor = torch.stack(frames)
    averaged_frame = frames_tensor.mean(dim=0)
    print(f"✅ Preprocessing completed for {len(frames)} frames.")
    return averaged_frame


def run_inference(shot_encoder, crn, shots):
    """ Run inference on all shots using kNN Matching.
        Also collects a representative (first) frame for each shot.
    """
    print("\n🚀 Running inference on shots...")
    all_scene_features = []
    rep_images = []  # store representative frames (PIL images)

    # Process shots in the order of their keys
    for shot_id, image_paths in shots.items():
        print(f"\n🔍 Processing Shot {shot_id} - {len(image_paths)} frames")
        input_tensor = preprocess_frames(image_paths).unsqueeze(0)
        # Save the first frame as the representative image
        rep_img = Image.open(image_paths[0]).convert("RGB")
        rep_images.append(rep_img)

        with torch.no_grad():
            shot_features = shot_encoder(input_tensor)
            shot_features = shot_features.unsqueeze(1)

            print(f"🔹 Shot {shot_id} - Input Shape to CRN: {shot_features.shape}")
            scene_features, _ = crn(shot_features)

            print(f"🔹 Shot {shot_id} - Output Shape from CRN: {scene_features.shape}")
            print(f"🔹 Scene Features Sample: {scene_features[0, 0, :10]}")
            all_scene_features.append(scene_features.squeeze(0).mean(dim=0).numpy())

    all_scene_features = np.array(all_scene_features)
    print("\n✅ Inference completed.")
    return all_scene_features, rep_images


def segment_scenes(features, threshold=0.85, k=5, temporal_window=15):
    """ Segment scenes using kNN with a fixed threshold """
    print(f"\n🎬 Segmenting scenes using threshold={threshold:.4f}...")
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

        while end_shot in used_shots:
            if end_shot + 1 < num_shots:
                end_shot += 1
            else:
                break

        if end_shot < num_shots - 1:
            similarity = cosine_similarity([features[end_shot]], [features[end_shot + 1]])[0][0]
            if similarity > threshold:
                end_shot += 1

        scenes.append((start_shot, min(end_shot, num_shots - 1)))
        used_shots.update(range(start_shot, end_shot + 1))
        print(f"🎬 Scene {len(scenes)}: Shot {start_shot} to Shot {end_shot}")
        i = end_shot + 1

    print("\n✅ Scene segmentation completed.")
    return scenes


def main():
    # CONFIG_PATH = "D:/bassl/bassl/finetune/ckpt/test/config.json"
    CONFIG_PATH = "C:/Users/aly/OneDrive/Documents/GitHub/movie-sizzling/AI/Scene/bassl/bassl/checkpoints/config.json"
    # CHECKPOINT_PATH = "D:/bassl/bassl/checkpoints/model-v1.ckpt"
    CHECKPOINT_PATH = "C:/Users/aly/OneDrive/Documents/GitHub/movie-sizzling/AI/Scene/bassl/bassl/checkpoints/model-v1.ckpt"

    # MOVIE_PATH = "D:/bassl/bassl/data/movienet/240P_frames/tt0093779/tt0093779"
    # MOVIE_PATH = "C:/Users/aly/OneDrive/Documents/GitHub/movie-sizzling/AI/Scene/bassl/bassl/data/movienet/240P_frames/tt0049730/tt0049730"
    MOVIE_PATH = "C:/Users/aly/OneDrive/Documents/GitHub/movie-sizzling/shots/heross"

    OUTPUT_DIR = "output_images"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    shots = {}
    # Process image filenames and group them by shot id
    for filename in sorted(os.listdir(MOVIE_PATH)):
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        shot_id = parts[1]
        img_path = os.path.join(MOVIE_PATH, filename)
        if shot_id not in shots:
            shots[shot_id] = []
        shots[shot_id].append(img_path)

    print("🔍 Shots Processed:", list(shots.keys()))

    shot_encoder, crn = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    # Run inference and collect representative images
    scene_features, rep_images = run_inference(shot_encoder, crn, shots)

    scenes_fixed = segment_scenes(scene_features, threshold=0.85)
    print(f"\n🔹 Total Scenes (Fixed Threshold 0.85): {len(scenes_fixed)}")

    # Save actual JPEG images for each shot in every scene
    print("\n🔹 Saving scene images:")
    for scene_idx, (start_shot, end_shot) in enumerate(scenes_fixed):
        print(f"\n🎬 Scene {scene_idx + 1}: Shots {start_shot} to {end_shot}")
        for shot_idx in range(start_shot, end_shot + 1):
            output_filename = os.path.join(OUTPUT_DIR, f"scene_{scene_idx + 1}_shot_{shot_idx}.jpg")
            rep_images[shot_idx].save(output_filename)
            print(f"Saved Shot {shot_idx} image to {output_filename}")


if __name__ == "__main__":
    main()