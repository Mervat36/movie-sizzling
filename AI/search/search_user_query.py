import os
import sys
import json
import torch
import clip
import subprocess
import cv2
from datetime import datetime

# CLI args: [captions_json, user_query, video_path]
captions_path = sys.argv[1]
query = sys.argv[2]
video_path = sys.argv[3]

# Load caption metadata
with open(captions_path, "r") as f:
    data = json.load(f)
metadata = data["shots_metadata"]

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Process query
query_embedding = model.encode_text(clip.tokenize([query]).to(device)).squeeze()
caption_results = []

# Match captions
for path, info in metadata.items():
    caption = info["caption"]
    caption_embedding = model.encode_text(clip.tokenize([caption]).to(device)).squeeze()
    score = torch.cosine_similarity(query_embedding, caption_embedding, dim=0).item()

    if score > 0.28:
        caption_results.append({
            "score": round(score, 3),
            "caption": caption,
            "start_time": info["start_time"],
            "end_time": info["end_time"],
            "image": path
        })

# Sort and remove duplicates by time
seen = set()
final_matches = []
for result in sorted(caption_results, key=lambda x: x["score"], reverse=True):
    key = (result["start_time"], result["end_time"])
    if key not in seen:
        seen.add(key)
        final_matches.append(result)

# Create folder for scene clips
scene_clips_folder = "output_scene_clips"
os.makedirs(scene_clips_folder, exist_ok=True)

# Initialize list to store scene video files
scene_video_files = []

def timestamp_to_seconds(ts):
    """Convert HH:MM:SS.MS to float seconds"""
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

# Open the video file using OpenCV
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process and cut scenes from video
for i, match in enumerate(final_matches):
    start = timestamp_to_seconds(match["start_time"])
    end = timestamp_to_seconds(match["end_time"])

    # Set the position to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))

    # Create a VideoWriter object to save the cut scene
    output_filename = os.path.join(scene_clips_folder, f"scene_{i:03}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

        # Check if we've reached the end of the scene
        if current_frame >= (end * fps):
            break

        # Write the frame to the output video file
        out.write(frame)

    # Release the video writer and continue
    out.release()

    # Save the scene metadata
    scene_video_files.append({
        "file": output_filename,
        "start_time": match["start_time"],
        "end_time": match["end_time"],
        "caption": match["caption"],
        "score": match["score"]
    })

# Release the video capture object
cap.release()

# Save JSON of matched scenes
with open("matched_scenes.json", "w") as f:
    json.dump(scene_video_files, f, indent=2)

print(f"✅ Finished. Total scenes: {len(scene_video_files)}")
