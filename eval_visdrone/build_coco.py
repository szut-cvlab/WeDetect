#!/usr/bin/env python
"""VisDrone YOLO txt -> COCO json + prompt 表。见 notes/2026-11-04-visdrone.md Step 1。"""
import json
import os
import os.path as osp
import glob

from PIL import Image

ROOT = "data/VisDrone"

ZH = ["行人", "人", "自行车", "汽车", "面包车", "卡车", "三轮车", "遮阳三轮车", "公交车", "摩托车"]
EN = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
      "awning-tricycle", "bus", "motor"]

CATS = ZH  # GT json 的 categories，也是 metainfo 类名

TEXTS_DIR = "data/texts"
TABLES = {
    "visdrone_zh_class_texts.json": ZH,       # 主跑：10 正类
    "visdrone_en_class_texts.json": EN,       # 中英对照：同 10 类换英文
}


def dump_coco(split, out_name, categories):
    images, annotations = [], []
    per_cls = [0] * len(ZH)
    ann_id = 1
    for img_id, label in enumerate(sorted(glob.glob(osp.join(ROOT, "labels", split, "*.txt"))), 1):
        stem = osp.basename(label)[:-4]
        img_path = osp.join(ROOT, "images", split, stem + ".jpg")
        if not osp.exists(img_path):
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": img_id, "file_name": f"{split}/{stem}.jpg",
                       "width": W, "height": H})
        with open(label) as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                cls = int(p[0])
                xc, yc, wn, hn = map(float, p[1:])
                per_cls[cls] += 1
                w, h = wn * W, hn * H
                x, y = xc * W - w / 2, yc * H - h / 2
                x = min(max(x, 0.0), W)
                y = min(max(y, 0.0), H)
                w, h = min(max(w, 1.0), W - x), min(max(h, 1.0), H - y)
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2), "iscrowd": 0,
                })
                ann_id += 1
    cats = categories
    coco = {
        "info": {"description": f"VisDrone2019-DET-{split} converted from YOLO labels"},
        "licenses": [], "images": images,
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cats)],
        "annotations": annotations,
    }
    out = osp.join(ROOT, "annotations", out_name)
    os.makedirs(osp.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f)
    print(f"[{out_name}] imgs={len(images)} boxes={len(annotations)} cats={len(cats)}")
    print("        per-class:", dict(zip(ZH, per_cls)))
    return len(images), len(annotations)


def dump_texts():
    os.makedirs(TEXTS_DIR, exist_ok=True)
    for name, words in TABLES.items():
        with open(osp.join(TEXTS_DIR, name), "w") as f:
            json.dump([[w] for w in words], f, ensure_ascii=False)
        print(f"[texts] {name}: {len(words)} 条")


if __name__ == "__main__":
    n = dump_coco("test", "visdrone_testdev_coco.json", categories=CATS)
    dump_texts()

    assert n == (1610, 75102), n
    assert len(ZH) == len(EN) == len(CATS) == 10
    print("OK")
