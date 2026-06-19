import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGES = ROOT / "data" / "raw" / "10k"
RAW_LABELS = ROOT / "data" / "raw" / "labels"

YOLO_ROOT = ROOT / "data" / "yolo"

IMG_W = 1280
IMG_H = 720

CLASS_MAP = {
    "car": 0,
    "person": 1,
    "traffic light": 2,
    "traffic sign": 3,
    "bike": 4,
    "bus": 5,
    "truck": 6,
    "rider": 7,
}


def convert_bbox(box):
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    xc = ((x1 + x2) / 2) / IMG_W
    yc = ((y1 + y2) / 2) / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H

    return xc, yc, w, h


pairs = []

for split in ["train", "val", "test"]:

    image_dir = RAW_IMAGES / split
    label_dir = RAW_LABELS / split

    images = {p.stem: p for p in image_dir.glob("*.jpg")}

    for json_file in label_dir.glob("*.json"):

        stem = json_file.stem

        if stem not in images:
            continue

        pairs.append(
            {
                "image": images[stem],
                "json": json_file
            }
        )

print(f"\nMatched pairs found: {len(pairs)}")

train_pairs, temp_pairs = train_test_split(
    pairs,
    test_size=0.30,
    random_state=42
)

val_pairs, test_pairs = train_test_split(
    temp_pairs,
    test_size=0.50,
    random_state=42
)

splits = {
    "train": train_pairs,
    "val": val_pairs,
    "test": test_pairs
}

for split_name, split_pairs in splits.items():

    img_out = YOLO_ROOT / "images" / split_name
    lbl_out = YOLO_ROOT / "labels" / split_name

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {split_name}: {len(split_pairs)}")

    for pair in tqdm(split_pairs):

        image_path = pair["image"]
        json_path = pair["json"]

        with open(json_path) as f:
            annotation = json.load(f)

        yolo_lines = []

        frames = annotation.get("frames", [])

        for frame in frames:

            for obj in frame.get("objects", []):

                category = obj.get("category")

                if category not in CLASS_MAP:
                    continue

                if "box2d" not in obj:
                    continue

                xc, yc, w, h = convert_bbox(
                    obj["box2d"]
                )

                yolo_lines.append(
                    f"{CLASS_MAP[category]} "
                    f"{xc:.6f} "
                    f"{yc:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )

        if not yolo_lines:
            continue

        shutil.copy2(
            image_path,
            img_out / image_path.name
        )

        with open(
            lbl_out / f"{image_path.stem}.txt",
            "w"
        ) as f:
            f.write("\n".join(yolo_lines))

print("\nDataset build complete.")