"""训练入口合并：FOD 集 -> fod.json，+ COD -> ovd.json（统一 id，只留标准字段）。

各集 json 的 category_id 生成时对应各自的词表快照，这里统一按类名
重映射到当前 vocab_merged.json 的 id，各集 json 本身不重跑。

image: id / file_name / width / height
ann:   id / image_id / category_id / bbox / area / iscrowd
附加字段（score/truncate/occlusion/ignore_regions 等）只在单集 json 里保留。
"""
import json
import os

COCO = "/mnt/sda1/datasets/drone/coco"
FOD = ["visdrone_trainval", "codrone", "dronevehicle", "hazydet", "atrumod",
       "mobdrone_train", "seadronessee"]
OUTS = {"fod": FOD, "ovd": FOD + ["cod"]}
VOCAB = json.load(open(f"{COCO}/vocab_merged.json"))
NAME2UID = {n: i for i, n in enumerate(VOCAB)}


def merge(names):
    images, anns, names_seen = [], [], set()
    for name in names:
        d = json.load(open(f"{COCO}/{name}.json"))
        old = {c["id"]: c["name"] for c in d["categories"]}
        miss = set(old.values()) - set(NAME2UID)
        assert not miss, f"{name}: categories not in vocab {sorted(miss)}"
        ib, ab = len(images), len(anns)
        for k, im in enumerate(d["images"]):
            images.append({"id": ib + k, "file_name": im["file_name"],
                           "width": im["width"], "height": im["height"]})
        for k, a in enumerate(d["annotations"]):
            anns.append({"id": ab + k, "image_id": ib + a["image_id"],
                         "category_id": NAME2UID[old[a["category_id"]]],
                         "bbox": a["bbox"], "area": a["area"],
                         "iscrowd": a.get("iscrowd", 0)})
        names_seen |= set(old.values())
    return images, anns, [{"id": NAME2UID[n], "name": n, "supercategory": ""}
                          for n in sorted(names_seen)]


for out, names in OUTS.items():
    images, anns, cats = merge(names)
    json.dump({"images": images, "annotations": anns, "categories": cats},
              open(f"{COCO}/{out}.json", "w"))
    json.dump([[c["name"]] for c in cats], open(f"{COCO}/{out}_class_texts.json", "w"),
              ensure_ascii=False)
    print(f"{out}.json: {len(images)} imgs / {len(anns)} anns / {len(cats)} cats")
    miss = sum(1 for im in images if not os.path.exists(f"/mnt/sda1/datasets/drone/{im['file_name']}"))
    print(f"  file_name 校验 missing={miss}")
