import os
import torch
import clip
import json
import pickle
import re
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
import spacy
from pymongo import MongoClient
from datetime import datetime

# ----------------------------
# Setup
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
yolo = YOLO("yolov8n.pt")
nlp = spacy.load("en_core_web_sm")


# ----------------------------
# Tag Extraction
# ----------------------------
place_keywords = {"restaurant", "kitchen", "hall", "room", "forest", "street", "car", "park", "school", "office", "building", "beach", "hospital"}

def extract_tags(text):
    doc = nlp(text)
    action, place = None, None
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
    return {"action": action, "place": place, "objects": objects}

# ----------------------------
# Load Cached Data
# ----------------------------
with open("clip_embeddings.pkl", "rb") as f:
    clip_cache = pickle.load(f)
scene_features = torch.stack(clip_cache["features"]).squeeze()
scene_paths = clip_cache["paths"]

with open("scene_metadata.json", "r") as f:
    scene_metadata = json.load(f)

# ----------------------------
# Search Execution
# ----------------------------
query = input("🔍 Enter your query: ").strip()
tags = extract_tags(query)

query_tokens = clip.tokenize([query]).to(device)
with torch.no_grad():
    query_embedding = model.encode_text(query_tokens).mean(dim=0)

similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), scene_features)
threshold = 0.25

results = []
for i, sim in enumerate(similarities):
    score = sim.item()
    if score < threshold:
        continue

    path = scene_paths[i]
    data = scene_metadata.get(path)
    if not data:
        continue

    caption = data["caption"]
    image_tags = data["tags"]
    yolo_objects = list(set(image_tags["objects"]))

    if not set(tags["objects"]).intersection(yolo_objects):
        continue

    cap_score = util.pytorch_cos_sim(
        sentence_model.encode(query, convert_to_tensor=True),
        sentence_model.encode(caption, convert_to_tensor=True)
    ).item()

    final_score = 0.7 * score + 0.3 * cap_score

    results.append({
        "image": os.path.basename(path),
        "clip_score": round(score, 3),
        "caption_score": round(cap_score, 3),
        "final_score": round(final_score, 3),
        "caption": caption,
        "tags": image_tags,
        "yolo_objects": yolo_objects,
        "start_time": data.get("start_time", ""),
        "end_time": data.get("end_time", "")
    })

results.sort(key=lambda x: x["final_score"], reverse=True)

# ----------------------------
# Save to Mongo
# ----------------------------
entry = {
    "userId": None,
    "result": {
        "query": query,
        "title": "search_results",
        "nlp_tags": tags,
        "matches": results,
        "createdAt": datetime.utcnow()
    },
    "createdAt": datetime.utcnow()
}

user_query_collection.insert_one(entry)
print(f"✅ Saved query with {len(results)} matches to MongoDB.")
