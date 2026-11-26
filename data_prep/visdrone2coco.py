#!/usr/bin/env python
"""VisDrone DET 标注 -> COCO JSON（统一词表 id）。

用法
    python3 data_prep/visdrone2coco.py                  # train+val -> visdrone_trainval.json
    python3 data_prep/visdrone2coco.py --split testdev  # test-dev -> visdrone_testdev.json

输入  VisDrone/images/{train,val,test}/*.jpg
      VisDrone/VisDrone2019-DET-{train,val,test-dev}/annotations/*.txt
      每行 8 字段：left,top,w,h,score,category,truncate,occlusion
输出  coco/visdrone_trainval.json / coco/visdrone_testdev.json
      类 1-10 写进 annotations，0 进 ignore_regions，11 进 others_regions（只留档，不进 GT）
      ignore 的 don't-care 由评测逻辑处理（自定义 eval：未匹配 GT 且落在 ignore 区域的检出不计 FP，
      VisDrone 官方 devkit 语义），不在 json 里做 crowd 化——全类复制 crowd ann 会让
      per-category 统计失真，训练侧误用还会变错误正样本。
      file_name 相对 /mnt/sda1/datasets/drone，全池共用 data_root
      class_texts 只在 trainval 时写，testdev 共用同一份
"""
import argparse
import json
import os
from collections import Counter

from PIL import Image

ROOT = "/mnt/sda1/datasets/drone/VisDrone"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning tricycle", "bus", "motorcycle"]
# (图像子目录, 标注后缀) / 输出名 / 是否写 class_texts
SPLITS = {
    "trainval": ([("train", "train"), ("val", "val")], "visdrone_trainval.json", True),
    "testdev": ([("test", "test-dev")], "visdrone_testdev.json", False),
}

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=SPLITS, default="trainval")
args = ap.parse_args()
pairs, out_name, write_texts = SPLITS[args.split]

VOCAB = json.load(open(f"{OUT_DIR}/vocab_merged.json"))
NAME2UID = {n: i for i, n in enumerate(VOCAB)}
cats = [{"id": NAME2UID[n], "name": n, "supercategory": ""} for n in CLASSES]

images, anns = [], []
stat = Counter()
for img_sub, ann_sub in pairs:
    img_dir = f"{ROOT}/images/{img_sub}"
    ann_dir = f"{ROOT}/VisDrone2019-DET-{ann_sub}/annotations"
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".jpg"):
            continue
        with Image.open(os.path.join(img_dir, fn)) as im:
            w, h = im.size
        iid = len(images)
        images.append({"id": iid, "file_name": f"VisDrone/images/{img_sub}/{fn}",
                       "width": w, "height": h,
                       "ignore_regions": [], "others_regions": []})
        ann_path = os.path.join(ann_dir, fn[:-4] + ".txt")
        if not os.path.exists(ann_path):
            stat["no_anno_file"] += 1
            continue
        for line in open(ann_path):
            p = [v for v in line.strip().split(",") if v != ""]
            if len(p) != 8:
                stat["bad_line"] += 1
                continue
            left, top, ww, hh, score, c, trunc, occ = map(int, p)
            stat["lines"] += 1
            if ww <= 0 or hh <= 0:
                stat["empty_box"] += 1
                continue
            if c == 0:
                stat["ignore"] += 1
                images[-1]["ignore_regions"].append([left, top, ww, hh])
                continue
            if c == 11:
                stat["others"] += 1
                images[-1]["others_regions"].append([left, top, ww, hh])
                continue
            if not 1 <= c <= len(CLASSES):
                stat["unknown_cat"] += 1
                continue
            x, y = max(0, left), max(0, top)
            ww2, hh2 = min(ww, w - x), min(hh, h - y)
            if ww2 <= 0 or hh2 <= 0:
                stat["clipped_out"] += 1
                continue
            if score == 0:
                stat["score0"] += 1
            anns.append({"id": len(anns) + 1, "image_id": iid,
                         "category_id": NAME2UID[CLASSES[c - 1]],
                         "bbox": [x, y, ww2, hh2], "area": ww2 * hh2, "iscrowd": 0,
                         "score": score, "truncate": trunc, "occlusion": occ})

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"images": images, "annotations": anns, "categories": cats},
          open(f"{OUT_DIR}/{out_name}", "w"))
if write_texts:
    json.dump([[c["name"]] for c in cats],
              open(f"{OUT_DIR}/visdrone_class_texts.json", "w"), ensure_ascii=False)

print(f"{args.split}: images={len(images)} boxes={len(anns)}")
for k, v in sorted(stat.items()):
    print(f"  {k:12s} {v}")
cnt = Counter(a["category_id"] for a in anns)
for n in CLASSES:
    print(f"  {n:16s} {cnt[NAME2UID[n]]}")
print("  truncate", Counter(a["truncate"] for a in anns if "truncate" in a))
print("  occlusion", Counter(a["occlusion"] for a in anns if "occlusion" in a))
side = [min(a["bbox"][2], a["bbox"][3]) for a in anns if "truncate" in a]
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    print(f"  edge [{lo},{hi}) = {sum(1 for s in side if lo <= s < hi)}")
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(DRONE_ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
