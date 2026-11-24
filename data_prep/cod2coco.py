"""COD 训练集转 COCO：all.jsonl -> cod.json + cod_class_texts.json（统一词表 id）。"""
import json

COCO_DIR = "/mnt/sda1/datasets/drone/coco"
SKYFIND = "/mnt/sda1/datasets/drone/SkyFind"
ALL_JSONL = f"{SKYFIND}/cod_uni_boxes_fodcoco_sam3_named_norm_all.jsonl"
WH_JSONL = f"{SKYFIND}/cod_uni_boxes_fodcoco.jsonl"  # 宽高来源
COD_RENAME = {"suv": "SUV", "tanker truck": "tank truck"}

vocab = json.load(open(f"{COCO_DIR}/vocab_merged.json"))
cat_id = {n: i for i, n in enumerate(vocab)}
assert set(COD_RENAME.values()) <= set(vocab)

# 尾类归并表：先归并/剔除，再查统一 id（见 notes/2026-11-07 §2.3）
TAIL = json.load(open(f"{COCO_DIR}/vocab_tail_map.json"))
DROP, MERGE = set(TAIL["drop"]), TAIL["merge"]


def map_label(label):
    n = COD_RENAME.get(label, label)
    return None if n in DROP else MERGE.get(n, n)

wh = {}
for line in open(WH_JSONL):
    r = json.loads(line)
    if "width" in r:  # 7 行缺失图（missing）无宽高无框
        wh[r["file"]] = (r["width"], r["height"])

boxes = {}
for line in open(ALL_JSONL):
    r = json.loads(line)
    boxes.setdefault(r["file"], []).append(r)

mapped = {}
for f, rs in boxes.items():
    kept = [(r, n) for r in rs if (n := map_label(r["label"]))]
    if kept:                      # 框全被尾类剔除的图不入池
        mapped[f] = kept

files = sorted(mapped)
images, annotations = [], []
for img_id, f in enumerate(files):
    w, h = wh[f]
    images.append({"id": img_id, "file_name": f"SkyFind/train/{f}", "width": w, "height": h})
    for r, name in mapped[f]:
        x, y, bw, bh = r["box"]
        annotations.append({
            "id": len(annotations), "image_id": img_id, "category_id": cat_id[name],
            "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0, "score": r["score"],
        })

used = sorted({n for rs in mapped.values() for _, n in rs})
categories = [{"id": cat_id[n], "name": n, "supercategory": ""} for n in used]
json.dump({"images": images, "annotations": annotations, "categories": categories},
          open(f"{COCO_DIR}/cod.json", "w"))
json.dump([[n] for n in used], open(f"{COCO_DIR}/cod_class_texts.json", "w"))
print(len(images), "images /", len(annotations), "boxes /", len(used), "classes")
