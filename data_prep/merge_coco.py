"""训练入口合并：5 集 -> fod.json，+ COD -> ovd.json（统一 id，只留标准字段）。

image: id / file_name / width / height
ann:   id / image_id / category_id / bbox / area / iscrowd
附加字段（score/truncate/occlusion/ignore_regions 等）只在单集 json 里保留。
"""
import json
import os

COCO = "/mnt/sda1/datasets/drone/coco"
FOD = ["visdrone_trainval", "codrone", "dronevehicle", "hazydet", "atrumod"]
OUTS = {"fod": FOD, "ovd": FOD + ["cod"]}


def merge(names):
    images, anns, cats = [], [], {}
    for name in names:
        d = json.load(open(f"{COCO}/{name}.json"))
        ib, ab = len(images), len(anns)
        for k, im in enumerate(d["images"]):
            images.append({"id": ib + k, "file_name": im["file_name"],
                           "width": im["width"], "height": im["height"]})
        for k, a in enumerate(d["annotations"]):
            anns.append({"id": ab + k, "image_id": ib + a["image_id"],
                         "category_id": a["category_id"], "bbox": a["bbox"],
                         "area": a["area"], "iscrowd": a.get("iscrowd", 0)})
        for c in d["categories"]:
            cats[c["name"]] = c["id"]
    return images, anns, [{"id": cats[n], "name": n, "supercategory": ""} for n in sorted(cats)]


for out, names in OUTS.items():
    images, anns, cats = merge(names)
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(f"{COCO}/{out}.json", "w"))
    json.dump([[c["name"]] for c in cats], open(f"{COCO}/{out}_class_texts.json", "w"),
              ensure_ascii=False)
    print(f"{out}.json: {len(images)} imgs / {len(anns)} anns / {len(cats)} cats")
    miss = sum(1 for im in images if not os.path.exists(f"/mnt/sda1/datasets/drone/{im['file_name']}"))
    print(f"  file_name 校验 missing={miss}")
