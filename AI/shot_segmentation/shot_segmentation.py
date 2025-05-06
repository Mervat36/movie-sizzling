import tensorflow as tf
import cv2
import numpy as np
import os
import sys
import json

# Add the full path to the `inference` folder directly
current_file_path = os.path.abspath(__file__)
inference_dir = os.path.join(os.path.dirname(current_file_path), "inference")
sys.path.insert(0, inference_dir)

# Now Python can directly import transnetv2.py from the added path
from transnetv2 import TransNetV2

# Initialize model
model = TransNetV2()
print("[TransNetV2] Model initialized successfully.")

def frame_to_time(frame_index, fps):
    seconds = frame_index / fps if fps > 0 else 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 3)
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"

def load_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (48, 27))
        frames.append(frame)
    cap.release()
    return np.array(frames)

def load_video_high_res(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360))  # reduce resolution
        frames.append(frame)
    cap.release()
    return frames


def save_shot_images(output_dir, movie_title, shot_index, shot_info):
    shot_folder = os.path.join(output_dir, movie_title)
    os.makedirs(shot_folder, exist_ok=True)

    start_path = os.path.join(shot_folder, f"shot_{shot_index:03}_start.jpg")
    middle_path = os.path.join(shot_folder, f"shot_{shot_index:03}_middle.jpg")
    end_path = os.path.join(shot_folder, f"shot_{shot_index:03}_end.jpg")

    cv2.imwrite(start_path, shot_info["start_frame"])
    cv2.imwrite(middle_path, shot_info["middle_frame"])
    cv2.imwrite(end_path, shot_info["end_frame"])

    return {
        "start": start_path,
        "middle": middle_path,
        "end": end_path
    }

def predict_and_handle_transitions(frames, fps, min_shot_length=20, hard_cut_threshold=0.5, gradual_threshold=0.3, min_gradual_duration=5):
    predictions = model.predict_frames(frames)
    transition_scores = predictions[1]

    hard_cuts = np.where(transition_scores > hard_cut_threshold)[0]
    gradual_transitions = []

    start = None
    for i in range(len(transition_scores)):
        if transition_scores[i] > gradual_threshold:
            if start is None:
                start = i
        elif start is not None:
            if (i - start) >= min_gradual_duration:
                gradual_transitions.append((start, i - 1))
            start = None

    formatted_hard_cuts = [
        (start, end - 1)
        for start, end in zip(hard_cuts[:-1], hard_cuts[1:])
        if (end - start) >= min_shot_length
    ]

    return formatted_hard_cuts + gradual_transitions

def main():
    if len(sys.argv) < 3:
        print("Usage: python shot_segmentation.py <video_local_path> <movie_title>")
        return

    local_video_path = sys.argv[1]
    movie_title = sys.argv[2]
    shots_dir = "shots"
    output_json_dir = "output"

    print("[INFO] Processing local video:", local_video_path)
    print("Movie title:", movie_title)

    original_frames = load_video_high_res(local_video_path)
    frames = load_video(local_video_path)
    fps = cv2.VideoCapture(local_video_path).get(cv2.CAP_PROP_FPS)

    transitions = predict_and_handle_transitions(frames, fps)

    json_output = {
        "movieTitle": movie_title,
        "videoPath": local_video_path,
        "fps": float(fps),
        "shots": []
    }

    for i, (start, end) in enumerate(transitions):
        if start < 0 or end >= len(original_frames) or start > end:
            continue
        middle = (start + end) // 2
        shot_info = {
            "start_frame": original_frames[start],
            "middle_frame": original_frames[middle],
            "end_frame": original_frames[end],
            "start_index": int(start),
            "middle_index": int(middle),
            "end_index": int(end)
        }

        # Save images and get paths
        image_paths = save_shot_images(shots_dir, movie_title, i + 1, shot_info)

        json_output["shots"].append({
            "shotNumber": i + 1,
            "startFrame": int(start),
            "middleFrame": int(middle),
            "endFrame": int(end),
            "startTime": frame_to_time(start, fps),
            "middleTime": frame_to_time(middle, fps),
            "endTime": frame_to_time(end, fps),
            "images": image_paths
        })

    os.makedirs(output_json_dir, exist_ok=True)
    json_path = os.path.join(output_json_dir, f"{movie_title}_shots.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Shot metadata saved to {json_path}")
    print(f"[INFO] Shot images saved to {os.path.join(shots_dir, movie_title)}")

if __name__ == "__main__":
    main()




# import tensorflow as tf
# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from inference.transnetv2 import TransNetV2  # cleaner import

# # Initialize the TransNetV2 model
# model = TransNetV2()
# print("[TransNetV2] Model initialized successfully.")

# def frame_to_time(frame_index, fps):
#     seconds = frame_index / fps if fps > 0 else 0
#     hours = int(seconds // 3600)
#     minutes = int((seconds % 3600) // 60)
#     seconds = round(seconds % 60, 3)
#     return f"{hours:02}:{minutes:02}:{seconds:06.3f}"

# def load_video(filename):
#     cap = cv2.VideoCapture(filename)
#     frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.resize(frame, (48, 27))
#         frames.append(frame)
#     cap.release()
#     return np.array(frames)

# def load_video_high_res(filename):
#     cap = cv2.VideoCapture(filename)
#     frames = []
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)
#     cap.release()
#     return frames

# def visualize_shot_frames(original_frames, shot_info, fps):
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     fig.suptitle(f"Shot Frames - {frame_to_time(shot_info['start_index'], fps)} to {frame_to_time(shot_info['end_index'], fps)}")

#     axs[0].imshow(cv2.cvtColor(original_frames[shot_info['start_index']], cv2.COLOR_BGR2RGB))
#     axs[0].set_title("Start")
#     axs[0].axis('off')

#     axs[1].imshow(cv2.cvtColor(original_frames[shot_info['middle_index']], cv2.COLOR_BGR2RGB))
#     axs[1].set_title("Middle")
#     axs[1].axis('off')

#     axs[2].imshow(cv2.cvtColor(original_frames[shot_info['end_index']], cv2.COLOR_BGR2RGB))
#     axs[2].set_title("End")
#     axs[2].axis('off')

#     plt.show()

# def predict_and_handle_transitions(frames, fps, min_shot_length=20, hard_cut_threshold=0.5, gradual_threshold=0.3, min_gradual_duration=5):
#     predictions = model.predict_frames(frames)
#     transition_scores = predictions[1]

#     hard_cuts = np.where(transition_scores > hard_cut_threshold)[0]
#     gradual_transitions = []

#     start = None
#     for i in range(len(transition_scores)):
#         if transition_scores[i] > gradual_threshold:
#             if start is None:
#                 start = i
#         elif start is not None:
#             if (i - start) >= min_gradual_duration:
#                 gradual_transitions.append((start, i - 1))
#             start = None

#     formatted_hard_cuts = [
#         (start, end - 1)
#         for start, end in zip(hard_cuts[:-1], hard_cuts[1:])
#         if (end - start) >= min_shot_length
#     ]

#     return formatted_hard_cuts + gradual_transitions

# def main():
#     video_paths = ['AI/shot_segmentation/test.mp4']

#     for video_path in video_paths:
#         print("\nProcessing Video:", video_path)
#         original_frames = load_video_high_res(video_path)
#         frames = load_video(video_path)
#         fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

#         transitions = predict_and_handle_transitions(frames, fps)

#         for start, end in transitions:
#             if start < 0 or end >= len(original_frames) or start > end:
#                 continue
#             middle = (start + end) // 2
#             shot_info = {
#                 "start_frame": original_frames[start],
#                 "middle_frame": original_frames[middle],
#                 "end_frame": original_frames[end],
#                 "start_index": start,
#                 "middle_index": middle,
#                 "end_index": end
#             }
#             visualize_shot_frames(original_frames, shot_info, fps)

# if __name__ == "__main__":
#     main()


