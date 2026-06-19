from pathlib import Path

for split in ["train", "val", "test"]:

    image_dir = Path(f"data/raw/10k/{split}")
    label_dir = Path(f"data/raw/labels/{split}")

    images = {p.stem for p in image_dir.glob("*.jpg")}
    labels = {p.stem for p in label_dir.glob("*.json")}

    matches = images & labels

    print(f"\n{split.upper()}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Matches: {len(matches)}")