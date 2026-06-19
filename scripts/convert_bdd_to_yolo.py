import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGES = ROOT / "data" / "raw" / "10k"
RAW_LABELS = ROOT / "data" / "raw" / "labels"

YOLO_ROOT = ROOT / "data" / "yolo"

IMG_WIDTH = 1280
IMG_HEIGHT = 720

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

# ============================================================
# HELPERS
# ============================================================

def convert_bbox(box):
    """
    Convert BDD100K box2d -> YOLO format
    """

    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    x_center = ((x1 + x2) / 2) / IMG_WIDTH
    y_center = ((y1 + y2) / 2) / IMG_HEIGHT

    width = (x2 - x1) / IMG_WIDTH
    height = (y2 - y1) / IMG_HEIGHT

    return x_center, y_center, width, height


def process_split(split):

    image_dir = RAW_IMAGES / split
    label_dir = RAW_LABELS / split

    out_img_dir = YOLO_ROOT / "images" / split
    out_lbl_dir = YOLO_ROOT / "labels" / split

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))

    print(f"\nProcessing {split}: {len(image_files)} images")

    kept_images = 0

    for image_path in tqdm(image_files):

        image_stem = image_path.stem

        json_path = label_dir / f"{image_stem}.json"

        if split == "val":
            if not json_path.exists():
                print(f"MISSING: {image_path.name}")
                print(f"EXPECTED: {json_path.name}")
                break

        if not json_path.exists():
            continue

        try:
            with open(json_path, "r") as f:
                annotation = json.load(f)

        except Exception as e:
            print(f"Error reading {json_path}: {e}")
            continue

        yolo_lines = []

        frames = annotation.get("frames", [])

        for frame in frames:

            objects = frame.get("objects", [])

            for obj in objects:

                category = obj.get("category")

                if category not in CLASS_MAP:
                    continue

                if "box2d" not in obj:
                    continue

                box = obj["box2d"]

                x_center, y_center, width, height = convert_bbox(box)

                line = (
                    f"{CLASS_MAP[category]} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{width:.6f} "
                    f"{height:.6f}"
                )

                yolo_lines.append(line)

        # Skip images without desired objects
        if len(yolo_lines) == 0:
            continue

        shutil.copy2(
            image_path,
            out_img_dir / image_path.name
        )

        with open(
            out_lbl_dir / f"{image_stem}.txt",
            "w"
        ) as f:
            f.write("\n".join(yolo_lines))

        kept_images += 1

    print(f"Kept {kept_images} images")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_split("train")
    process_split("val")
    process_split("test")

    print("\nConversion complete.")