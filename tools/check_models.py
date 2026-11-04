"""Step 1 check. Does Muse Spark read a label photo, and does SAM 3.1 count boxes?

Run from the repo root:
    python -m tools.check_models LABEL_PHOTO BOXES_PHOTO
"""
import sys
import time

from src.meta_api import read_label, segment

label_photo, boxes_photo = sys.argv[1], sys.argv[2]

start = time.time()
fields, usage = read_label(label_photo)
print(f"Muse Spark, {time.time() - start:.1f} s, {usage['total_tokens']} tokens")
for name, value in fields.items():
    print(f"  {name}: {value!r}")

start = time.time()
boxes = segment(boxes_photo, "cardboard box")
print(f"SAM 3.1, {time.time() - start:.1f} s, {len(boxes)} boxes found")
for box in boxes:
    print(f"  {box}")
