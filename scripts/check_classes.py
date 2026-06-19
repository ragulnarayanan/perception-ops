from pathlib import Path
from collections import Counter

counter = Counter()

for txt_file in Path("data/yolo/labels/train").glob("*.txt"):

    with open(txt_file) as f:

        for line in f:

            cls = int(line.split()[0])

            counter[cls] += 1

print(counter)