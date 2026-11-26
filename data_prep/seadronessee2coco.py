#!/usr/bin/env python
"""SeaDronesSee Object Detection v2 -> COCO JSON（训练池，统一词表 id）。

用法
    python3 data_prep/seadronessee2coco.py  # -> seadronessee.json（train+val 全量入 FOD）

输入  SeaDronesSee/data/images/{train,val}/shard_*/{*.jpg,metadata.jsonl}
      metadata.jsonl 每行 {file_name, objects:{bbox:[[x,y,w,h]...], category:[name...]}}
      test 3,750 图无标注，不生成。
类别 5 类统一词表：swimmer 152、boat 20、jetski 176、life saving appliances 177、buoy 174。
"""
import json
import os
from collections import Counter

from PIL import Image

ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{ROOT}/coco"
OUT_NAME = "seadronessee"
CLASSES = ["swimmer", "boat", "jetski", "life saving appliances", "buoy"]
RAW2NAME = {"life_saving_appliances": "life saving appliances"}
SPLITS = ["train", "val"]

vocab = json.load(open(f"{OUT_DIR}/vocab_merged.json"))
cat_id = {n: i for i, n in enumerate(vocab)}
assert set(CLASSES) <= set(vocab), set(CLASSES) - set(vocab)

images, anns = [], []
stat = Counter()
for sp in SPLITS:
    SRC = f"{ROOT}/SeaDronesSee/data/images/{sp}"
    for shard in sorted(os.listdir(SRC)):
        sd = os.path.join(SRC, shard)
        if not os.path.isdir(sd):
            continue
        mj = os.path.join(sd, "metadata.jsonl")
        if not os.path.exists(mj):
            continue
        for line in open(mj):
            if not line.strip():
                continue
            r = json.loads(line)
            p = os.path.join(sd, r["file_name"])
            with Image.open(p) as im:
                w, h = im.size
            iid = len(images)
            images.append({"id": iid,
                           "file_name": f"SeaDronesSee/data/images/{sp}/{shard}/{r['file_name']}",
                           "width": w, "height": h})
            bbs = r.get("objects", {}).get("bbox", [])
            cats = r.get("objects", {}).get("category", [])
            for b, c in zip(bbs, cats):
                c = RAW2NAME.get(c, c)
                if c not in cat_id:
                    stat["unknown_cat"] += 1
                    continue
                x, y, bw, bh = b
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cat_id[c],
                             "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
        stat[f"images_{sp}"] += 1

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"images": images, "annotations": anns,
           "categories": [{"id": cat_id[n], "name": n, "supercategory": ""} for n in CLASSES]},
          open(f"{OUT_DIR}/{OUT_NAME}.json", "w"))
json.dump([[n] for n in CLASSES],
          open(f"{OUT_DIR}/{OUT_NAME}_class_texts.json", "w"), ensure_ascii=False)

print(f"{OUT_NAME}.json: images={len(images)} boxes={len(anns)} ({stat['images_train']} train + {stat['images_val']} val)")
for n in CLASSES:
    print(f"  {n:24s} {sum(1 for a in anns if a['category_id']==cat_id[n])}")
for k, v in sorted(stat.items()):
    if k.startswith("unknown"):
        print(f"  {k:12s} {v}")
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
