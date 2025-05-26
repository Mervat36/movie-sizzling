import os
import sys
import json
import torch
import clip
import cv2
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from nltk.corpus import wordnet

# Helper functions
def timestamp_to_seconds(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

def clean_text(text):
    import re
    return re.sub(r'[^a-zA-Z0-9 ]', '', text.lower()).strip()

def expand_query(text):
    words = clean_text(text).split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace('_', ' '))
    return list(expanded)

# Custom synonyms to manually boost weak NLP pairs
custom_synonyms = {
    "guy": "man",
    "vehicle": "car",
    "automobile": "car",
    "lady": "woman",
    "kid": "child",
    "road": "street",
    "truck": "car",
    "boy": "man"
}

def apply_custom_synonym_boost(query_tokens, caption_tokens):
    for q in query_tokens:
        for c in caption_tokens:
            if custom_synonyms.get(q) == c or custom_synonyms.get(c) == q:
                return 0.1  # add bonus boost
    return 0.0

# CLI args
captions_path = sys.argv[1]
query = sys.argv[2]
video_path = sys.argv[3]

# Load data
with open(captions_path, "r") as f:
    data = json.load(f)
metadata = data["shots_metadata"]

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Expand query and encode
expanded_queries = expand_query(query)
expanded_embeddings = sentence_model.encode(expanded_queries, convert_to_tensor=True)
query_tokens = clean_text(query).split()

caption_results = []

# Match captions
for path, info in metadata.items():
    caption = info["caption"]
    caption_clean = clean_text(caption)
    caption_embedding = sentence_model.encode(caption_clean, convert_to_tensor=True)
    caption_tokens = caption_clean.split()

    # Highest semantic similarity
    max_semantic_score = max([util.cos_sim(query_emb, caption_embedding).item() for query_emb in expanded_embeddings])
    fuzzy_score = fuzz.token_set_ratio(clean_text(query), caption_clean) / 100.0
    synonym_boost = apply_custom_synonym_boost(query_tokens, caption_tokens)

    combined_score = 0.65 * max_semantic_score + 0.25 * fuzzy_score + synonym_boost

    if combined_score > 0.3:
        caption_results.append({
            "score": round(combined_score, 3),
            "caption": caption,
            "start_time": info["start_time"],
            "end_time": info["end_time"],
            "image": path
        })

# Sort and filter duplicates
seen_paths = set()
final_matches = []
for result in sorted(caption_results, key=lambda x: timestamp_to_seconds(x["start_time"])):
    video_file = os.path.basename(result["image"]).split("_shot")[0]
    if video_file not in seen_paths:
        seen_paths.add(video_file)
        final_matches.append(result)

# Create output folder
scene_clips_folder = "output_scene_clips"
os.makedirs(scene_clips_folder, exist_ok=True)

# Optionally print results
import json
print(json.dumps(final_matches))

