"""构建 COD + FOD 统一词表 -> vocab_merged.json（0-based，按名排序固定 id）。"""
import json

COCO_DIR = "/mnt/sda1/datasets/drone/coco"
SKYFIND = "/mnt/sda1/datasets/drone/SkyFind"

COD_RENAME = {"suv": "SUV", "tanker truck": "tank truck"}
FOD_RENAME = {"motor": "motorcycle"}
COD_CLASSES_FILE = f"{SKYFIND}/cod_uni_boxes_fodcoco_sam3_named_norm_all_classes.jsonl"
FOD_FILES = ["visdrone_trainval", "codrone", "dronevehicle", "hazydet", "atrumod",
             "mobdrone_train", "seadronessee"]

# 尾类归并表：drop 剔除，merge 并入上位类（1-4 框的类，见 notes/2026-11-07 §2.3）
TAIL = json.load(open(f"{COCO_DIR}/vocab_tail_map.json"))
DROP, MERGE = set(TAIL["drop"]), TAIL["merge"]


def collapse(names):
    return {MERGE.get(n, n) for n in names if n not in DROP}


cod = collapse({COD_RENAME.get(n, n) for n in json.load(open(COD_CLASSES_FILE))})
fod = set()
for k in FOD_FILES:
    d = json.load(open(f"{COCO_DIR}/{k}.json"))
    fod |= {c["name"] for c in d["categories"]}
fod = collapse({FOD_RENAME.get(n, n) for n in fod})

vocab = sorted(cod | fod)
assert not (set(MERGE.values()) - set(vocab)), set(MERGE.values()) - set(vocab)
json.dump(vocab, open(f"{COCO_DIR}/vocab_merged.json", "w"), indent=1)
print(len(vocab), "classes ->", f"{COCO_DIR}/vocab_merged.json")
