#!/usr/bin/env python
"""MOBDrone -> COCO JSON（训练池 / 评测集，统一词表 id）。

用法
    python3 data_prep/mobdrone2coco.py --split train   # DJI_0915 -> mobdrone_train.json（入 FOD）
    python3 data_prep/mobdrone2coco.py --split test    # DJI_0804 -> mobdrone_test.json（评测）

输入  MOBDrone/annotations_5_custom_classes.json（126,170 图 / 181,689 框）
      MOBDrone/frames/*.PNG（9,128 张 k=10 有框帧，文件名 = 真实帧号，与标注 file_name 一致）
类映射 person/boat/surfboard 用原 id；wood→driftwood、life_buoy→lifebuoy（追加类）。
train 段 surfboard/lifebuoy 零框，但 categories 与 class_texts 统一 5 类，评测时共用。
"""
import argparse
import json
import os

ROOT = "/mnt/sda1/datasets/drone"
DRONE = f"{ROOT}/MOBDrone"
OUT_DIR = f"{ROOT}/coco"
RAW2NAME = {"person": "person", "boat": "boat", "surfboard": "surfboard",
            "wood": "driftwood", "life_buoy": "lifebuoy"}
CLASSES = ["person", "boat", "surfboard", "driftwood", "lifebuoy"]
SPLITS = {"train": ("DJI_0915", "mobdrone_train"), "test": ("DJI_0804", "mobdrone_test")}

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=SPLITS, required=True)
args = ap.parse_args()
prefix, out_name = SPLITS[args.split]

vocab = json.load(open(f"{OUT_DIR}/vocab_merged.json"))
cat_id = {n: i for i, n in enumerate(vocab)}
assert set(CLASSES) <= set(vocab), set(CLASSES) - set(vocab)
raw_id2name = {c["id"]: c["name"] for c in json.load(
    open(f"{DRONE}/annotations_5_custom_classes.json"))["categories"]}

files = {f for f in os.listdir(f"{DRONE}/frames")
         if f.endswith(".PNG") and f.startswith(prefix)}
src = json.load(open(f"{DRONE}/annotations_5_custom_classes.json"))
keep = {im["id"] for im in src["images"] if im["file_name"] in files}

images, anns = [], []
old2new = {}  # 官方原始 image id -> 0-based 连续 id（merge 偏移依赖连续性）
for im in src["images"]:
    if im["id"] in keep:
        old2new[im["id"]] = len(images)
        images.append({"id": len(images), "file_name": f"MOBDrone/frames/{im['file_name']}",
                       "width": im["width"], "height": im["height"]})
for a in src["annotations"]:
    name = RAW2NAME.get(raw_id2name[a["category_id"]])
    if a["image_id"] not in keep or name is None:
        continue
    x, y, w, h = a["bbox"]
    anns.append({"id": len(anns) + 1, "image_id": old2new[a["image_id"]],
                 "category_id": cat_id[name], "bbox": [x, y, w, h],
                 "area": w * h, "iscrowd": 0})

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"images": images, "annotations": anns,
           "categories": [{"id": cat_id[n], "name": n, "supercategory": ""} for n in CLASSES]},
          open(f"{OUT_DIR}/{out_name}.json", "w"))
if args.split == "train":
    json.dump([[n] for n in CLASSES],
              open(f"{OUT_DIR}/{out_name}_class_texts.json", "w"), ensure_ascii=False)

print(f"{args.split}: images={len(images)} boxes={len(anns)}")
for n in CLASSES:
    print(f"  {n:24s} {sum(1 for a in anns if a['category_id']==cat_id[n])}")
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
