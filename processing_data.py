
import os
import re
import random
import cv2 as cv
import torch

from tqdm import tqdm
from collections import defaultdict
from facenet_pytorch import MTCNN
from sklearn.model_selection import train_test_split




DATASET_PATH = "/DATA/deeksha/datasets/deep_dataset"
OUTPUT_DIR = "/DATA/deeksha/datasets/deep_dataset/Processed_data"

FRAME_SKIP = 50
FACE_CONF_THRESHOLD = 0.90
FACE_SIZE = (224, 224)
RANDOM_SEED = 42

DATASET_FOLDERS = [
    "FaceShifter",
    "Deepfakes",
    "DeepFakeDetection",
    "NeuralTextures",
    "Face2Face",
    "FaceSwap",
    "original",
]

random.seed(RANDOM_SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

mtcnn = MTCNN(keep_all=True, device=device)



# Output dir


for split in ["train", "val", "test"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)




def get_source_id(video_name: str) -> str:
 
    stem = os.path.splitext(video_name)[0]
    match = re.match(r"(\d+)", stem)
    return match.group(1) if match else stem


all_videos = []

for folder_name in DATASET_FOLDERS:
    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        print(f"[warn] missing folder, skipping: {folder_path}")
        continue

    label = "real" if folder_name == "original" else "fake"

    for video in os.listdir(folder_path):
        if video.lower().endswith((".mp4", ".avi", ".mov")):
            all_videos.append({
                "folder": folder_name,
                "video": video,
                "label": label,
                "source_id": get_source_id(video),
            })

print("Total videos found:", len(all_videos))

if not all_videos:
    raise RuntimeError(
        f"No videos found under {DATASET_PATH}. Check DATASET_PATH / folder names."
    )


id_to_labels = defaultdict(set)
for v in all_videos:
    id_to_labels[v["source_id"]].add(v["label"])

unique_ids = sorted(id_to_labels.keys())
id_strat_labels = [
    "fake" if "fake" in id_to_labels[sid] else "real"
    for sid in unique_ids
]

train_ids, temp_ids, train_strat, temp_strat = train_test_split(
    unique_ids,
    id_strat_labels,
    test_size=0.30,
    stratify=id_strat_labels,
    random_state=RANDOM_SEED,
)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=2 / 3,
    stratify=temp_strat,
    random_state=RANDOM_SEED,
)

train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

train_videos = [v for v in all_videos if v["source_id"] in train_ids]
val_videos = [v for v in all_videos if v["source_id"] in val_ids]
test_videos = [v for v in all_videos if v["source_id"] in test_ids]

print(f"Unique source identities: {len(unique_ids)}")
print(f"Train videos: {len(train_videos)}  (ids: {len(train_ids)})")
print(f"Val videos:   {len(val_videos)}  (ids: {len(val_ids)})")
print(f"Test videos:  {len(test_videos)}  (ids: {len(test_ids)})")



# Face extraction


def process_video_list(videos, split_name):
    saved_counts = {"real": 0, "fake": 0}
    skipped_videos = 0

    for item in tqdm(videos, desc=f"[{split_name}]"):
        folder_name = item["folder"]
        video_name = item["video"]
        label = item["label"]
        video_stem = os.path.splitext(video_name)[0]

        video_path = os.path.join(DATASET_PATH, folder_name, video_name)

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[warn] could not open video, skipping: {video_path}")
            skipped_videos += 1
            continue

        frame_id = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_id += 1
            if frame_id % FRAME_SKIP != 0:
                continue

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            try:
                boxes, probs = mtcnn.detect(rgb)
            except Exception as e:
                print(f"[warn] detection failed on {video_path} frame {frame_id}: {e}")
                continue

            if boxes is None:
                continue

            h, w = frame.shape[:2]

            for face_idx, (box, prob) in enumerate(zip(boxes, probs)):
                if prob is None or prob < FACE_CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv.resize(face, FACE_SIZE)


                save_name = (
                    f"{folder_name}_{video_stem}"
                    f"_frame{frame_id}_face{face_idx}.jpg"
                )
                save_path = os.path.join(OUTPUT_DIR, split_name, label, save_name)

                cv.imwrite(save_path, face)
                saved_counts[label] += 1

        cap.release()

    print(
        f"{split_name}: {saved_counts['real']} real faces, "
        f"{saved_counts['fake']} fake faces saved "
        f"({skipped_videos} videos skipped/unreadable)"
    )
    return saved_counts


if __name__ == "__main__":
    train_counts = process_video_list(train_videos, "train")
    val_counts = process_video_list(val_videos, "val")
    test_counts = process_video_list(test_videos, "test")

    print("\n=== Summary ===")
    for name, counts in [
        ("train", train_counts),
        ("val", val_counts),
        ("test", test_counts),
    ]:
        total = counts["real"] + counts["fake"]
        ratio = counts["fake"] / max(counts["real"], 1)
        print(
            f"{name:5s}: real={counts['real']:6d}  fake={counts['fake']:6d}  "
            f"total={total:6d}  fake:real ratio={ratio:.2f}"
        )

    print("\nDone!")