#!/usr/bin/env python3
"""
Smart Scene Detection - Automatically finds optimal parameters for each movie
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from test_movie import MovieSceneTestDataset, LGSSModel

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

def group_scenes(shot_ids, boundaries):
    """Group shots into scenes"""
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
    
    return scenes

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
    print(f"🎬 Smart Scene Detection for: {movie_folder}")
    print("=" * 60)
    
    # Get confidence scores
    confidences, shot_ids = get_confidence_scores(movie_folder, model_path)
    scores = [conf for _, conf in confidences]
    
    print(f"📊 Analyzing {len(shot_ids)} shots")
    print(f"   Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"   Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    
    # Define parameter ranges to test
    thresholds = np.arange(0.05, 0.8, 0.05)  # 0.05 to 0.75 in steps of 0.05 (lower start)
    gaps = [1, 2, 3, 4, 5]
    
    best_config = None
    best_quality = -1
    results = []
    
    print(f"\n🧪 Testing {len(thresholds)} thresholds × {len(gaps)} gaps = {len(thresholds) * len(gaps)} configurations...")
    
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
    print(f"\n🏆 Top 5 configurations:")
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
        print("❌ No valid configuration found!")
        return None
    
    print(f"\n✅ Best configuration selected:")
    print(f"   Threshold: {best_config['threshold']:.3f}")
    print(f"   Min gap: {best_config['gap']}")
    print(f"   Quality score: {best_config['quality']:.1f}")
    print(f"   Number of scenes: {best_config['num_scenes']}")
    
    # Show scene breakdown
    print(f"\n📽️  Scene breakdown:")
    for i, scene in enumerate(best_config['scenes'], 1):
        if scene:
            print(f"   Scene {i}: shots {scene[0]} to {scene[-1]} ({len(scene)} shots)")
    
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
        
        print(f"\n💾 Results saved to: smart_detection_results.json")
    
    return best_config

def batch_process_movies(movie_folders, model_path):
    """Process multiple movies with smart detection"""
    print(f"🎬 Batch Processing {len(movie_folders)} movies")
    print("=" * 60)
    
    results = {}
    
    for folder in movie_folders:
        if os.path.exists(folder):
            print(f"\n📁 Processing: {folder}")
            try:
                result = apply_smart_detection(folder, model_path, save_results=False)
                results[folder] = result
            except Exception as e:
                print(f"❌ Error processing {folder}: {e}")
                results[folder] = None
        else:
            print(f"❌ Folder not found: {folder}")
            results[folder] = None
    
    # Save batch results
    with open("batch_detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Batch results saved to: batch_detection_results.json")
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