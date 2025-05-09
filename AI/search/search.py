import os
import sys
import torch
import clip
import pickle
import json
from PIL import Image
from tqdm import tqdm
import spacy
from transformers import BlipProcessor, BlipForConditionalGeneration
import re

# -----------------------------
# NLP Setup
# -----------------------------
nlp = spacy.load("en_core_web_sm")

place_keywords = {
    "restaurant", "kitchen", "hall", "room", "forest", "street", "car", "park", "school", "office", "building", "beach", "hospital"
}

def extract_tags(text):
    doc = nlp(text)
    action = None
    place = None
    objects = []

    for token in doc:
        if token.pos_ == "VERB" and action is None:
            action = token.lemma_
        if token.dep_ == "pobj" and token.head.text in ["in", "on", "at"]:
            place = token.text
        if token.pos_ == "NOUN":
            if token.text.lower() in place_keywords:
                place = token.text
            else:
                objects.append(token.text)

    return {
        "action": action,
        "place": place,
        "objects": objects
    }

# -----------------------------
# Device Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# BLIP Setup (Captioning)
# -----------------------------
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def caption_and_extract(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = blip_model.generate(**inputs)
        caption = blip_processor.decode(output[0], skip_special_tokens=True)
    tags = extract_tags(caption)
    return caption, tags

# -----------------------------
# Receive args
# -----------------------------
if len(sys.argv) != 3:
    print(" Usage: python search.py <scene_image_folder> <scene_metadata_json>")
    exit(1)

image_folder = sys.argv[1]
metadata_file = sys.argv[2]
embedding_file = "clip_embeddings.pkl"

scene_features = []
scene_paths = []
clip_metadata = {}
scene_captions = []

with open(metadata_file, "r") as f:
    scene_json = json.load(f)

movie_name = scene_json.get("title", "unknown_movie").strip().replace(" ", "_")
scene_metadata_output = f"{movie_name}_captions.json"
scene_captions_file = f"{movie_name}_scene_summaries.json"

print("Encoding images from scene shot images:", image_folder)

scene_dict = {scene["scene_id"]: scene for scene in scene_json.get("scenes", [])}

all_files = os.listdir(image_folder)
scene_shot_map = {}

for filename in all_files:
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    match = re.match(r"(scene_\d+)_shot(\d+)_\w+", filename)
    if match:
        scene_id = match.group(1)
        shot_id = match.group(2)
        key = f"{scene_id}_shot{shot_id}"
        scene_shot_map.setdefault(scene_id, {}).setdefault(key, []).append(filename)

for scene_id, shot_dict in tqdm(scene_shot_map.items()):
    scene = scene_dict.get(scene_id)
    if not scene:
        print(f" Scene ID {scene_id} not found in metadata JSON.")
        continue

    scene_start_time = scene.get("start_time")
    scene_end_time = scene.get("end_time")

    all_captions = []
    for shot_key, files in sorted(shot_dict.items()):
        shot_captions = []
        for filename in sorted(files):
            path = os.path.join(image_folder, filename)
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)

            scene_features.append(embedding)
            scene_paths.append(path)

            caption, tags = caption_and_extract(path)

            clip_metadata[path] = {
                "caption": caption,
                "tags": tags,
                "start_time": scene_start_time,
                "end_time": scene_end_time
            }

            shot_captions.append(caption)

        all_captions.extend(shot_captions)

    scene_captions.append({
        "scene_id": scene_id,
        "start_time": scene_start_time,
        "end_time": scene_end_time,
        "start_shot": scene.get("start_shot"),
        "end_shot": scene.get("end_shot"),
        "captions": all_captions
    })


# Save clip embeddings
with open(embedding_file, "wb") as f:
    pickle.dump({
        "features": scene_features,
        "paths": scene_paths
    }, f)

# Save caption + tags for every shot image (renamed file)
with open(scene_metadata_output, "w") as f:
    json.dump({
        "movie_name": movie_name,
        "shots_metadata": clip_metadata
    }, f, indent=2)

# Save scene-level summaries with timestamps
with open(scene_captions_file, "w") as f:
    json.dump(scene_captions, f, indent=2)

scene_features = torch.stack(scene_features).squeeze()
print(f" Preprocessing complete. Metadata saved to {scene_metadata_output} and {scene_captions_file}")
