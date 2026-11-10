#!/usr/bin/env python3
"""五集合并为 FOD objectness 训练集：category_id 全并 0，类名 ['object']。

输入  coco/ 下五份标注（>1600 的集取 tiles 版，见 notes/2026-11-07-uav-ovd-data.md §2.1.0）
输出  coco/fod_objectness.json（file_name 相对 /mnt/sda1/datasets/drone，images/anns 重编 id）

运行  python3 data_prep/fod_objectness.py
"""
import json
import os

ROOT = "/mnt/sda1/datasets/drone"
SOURCES = ["visdrone_trainval_tiles.json", "dronevehicle.json",
           "hazydet_tiles.json", "codrone_tiles.json", "atrumod.json"]

images, anns = [], []
for src in SOURCES:
    d = json.load(open(f"{ROOT}/coco/{src}"))
    iid_base, aid_base = len(images), len(anns)
    remap = {im["id"]: iid_base + k for k, im in enumerate(d["images"])}
    for k, im in enumerate(d["images"]):
        im["id"] = iid_base + k
        images.append(im)
    for k, an in enumerate(d["annotations"]):
        an["id"] = aid_base + k
        an["image_id"] = remap[an["image_id"]]
        an["category_id"] = 0
        anns.append(an)
    print(f"{src}: {len(d['images'])} 图 {len(d['annotations'])} 框")

out = {"images": images, "annotations": anns,
       "categories": [{"id": 0, "name": "object", "supercategory": "object"}]}
json.dump(out, open(f"{ROOT}/coco/fod_objectness.json", "w"))
print(f"合计 {len(images)} 图 {len(anns)} 框 -> coco/fod_objectness.json")

miss = [im["file_name"] for im in images if not os.path.exists(f"{ROOT}/{im['file_name']}")]
print(f"file_name 校验 missing={len(miss)}", miss[:3] if miss else "")
