#!/usr/bin/env python
"""训练池切验证集：ovd.json -> ovd_train.json + ovd_val.json（约 3,000 图，防泄漏分组）。

val 配额（两层）：
    1. 类覆盖配额：每类抽 min(3, 全语料图数-1) 张图进 val（组内独占类拉最小组），
       只在全语料出现 1 图的类留在 train（移走即失唯一正样本）
    2. 按源比例填充至约 3,000 图（分组源按组为原子，贪心选小组防超发）
分组键（同组不跨 train/val）：
    CODrone   文件名去 _frame_<n> 后缀 = 场景
    MOBDrone  文件名去 _<6位帧号> 后缀 = 视频流
    其余（近重复已上游 dHash 剔除）按图随机
seed 固定可复现。
"""
import json
import random
import re
from collections import defaultdict
from collections import Counter
from pathlib import Path

COCO = Path("/mnt/sda1/datasets/drone/coco")
SRC = "ovd"
TARGET_VAL = 3000
QUOTA = 3
SEED = 0


def group_key(fn):
    """返回组键；None 表示该源按图随机。"""
    stem = Path(fn).stem
    if fn.startswith("CODrone/"):
        return re.sub(r"_frame_\d+$", "", stem)
    if fn.startswith("MOBDrone/"):
        return re.sub(r"_[0-9]{6}$", "", stem)
    return None


d = json.load(open(COCO / f"{SRC}.json"))
texts = json.load(open(COCO / f"{SRC}_class_texts.json"))

imgs = d["images"]
im_by_id = {im["id"]: im for im in imgs}
id2src = {im["id"]: im["file_name"].split("/")[0] for im in imgs}

# 分组：key = (源, 组键)；按图源的组键唯一
groups = defaultdict(list)
for im in imgs:
    k = group_key(im["file_name"])
    if k is None:
        k = f"solo:{im['file_name']}"
    groups[(im["file_name"].split("/")[0], k)].append(im["id"])
gsize = {g: len(v) for g, v in groups.items()}
src_of_group = {g: g[0] for g in groups}

rng = random.Random(SEED)
val_ids = set()

# ---- 1. 类覆盖配额：稀有类先挑，优先按图源（无组拖拽），分组源挑最小组 ----
cat_imgs = defaultdict(set)
for a in d["annotations"]:
    cat_imgs[a["category_id"]].add(a["image_id"])


def add_group(g):
    val_ids.update(groups[g])


for cid, cimgs in sorted(cat_imgs.items(), key=lambda x: len(x[1])):
    need = min(QUOTA, max(0, len(cimgs) - 1))
    if need == 0:
        continue
    solo = [i for i in cimgs if group_key(im_by_id[i]["file_name"]) is None]
    rng.shuffle(solo)
    for i in solo:
        if need == 0:
            break
        if i not in val_ids:
            val_ids.add(i)
            need -= 1
    if need > 0:  # 不足则从分组源拉最小组
        gcands = defaultdict(set)
        for i in cimgs:
            k = group_key(im_by_id[i]["file_name"])
            if k is not None:
                gcands[(id2src[i], k)].add(i)
        for g in sorted(gcands, key=lambda g: (gsize[g], rng.random())):
            have = len(gcands[g] & val_ids)
            if have >= need:
                break
            add_group(g)
            need -= len(gcands[g] & val_ids) - have

# ---- 2. 按源比例填充至约 TARGET_VAL ----
src_total = Counter(id2src[i] for i in im_by_id)
val_src_now = Counter(id2src[i] for i in val_ids)
for src, total in sorted(src_total.items()):
    target = round(TARGET_VAL * total / len(imgs))
    deficit = target - val_src_now[src]
    if deficit <= 0:
        continue
    glist = sorted((g for g in groups if src_of_group[g] == src and not (set(groups[g]) & val_ids)),
                   key=lambda g: g[0])
    grouped = group_key(im_by_id[groups[glist[0]][0]]["file_name"]) is not None if glist else False
    if grouped:
        rng.shuffle(glist)
        got = 0
        while got < deficit and glist:
            g = glist.pop(rng.randrange(len(glist)))
            add_group(g)
            got += gsize[g]
    else:
        pool = [i for g in glist for i in groups[g] if i not in val_ids]
        val_ids.update(rng.sample(pool, min(deficit, len(pool))))

# ---- 校验与落盘 ----
for g in groups:
    nv = len(set(groups[g]) & val_ids)
    assert nv == 0 or nv == gsize[g], f"组被拆散: {g}"

missing_cov = [c for c in cat_imgs if not (cat_imgs[c] & val_ids)]
bad = [c for c in missing_cov if len(cat_imgs[c]) > 1]
assert not bad, f">=2 图的类不应缺 val 覆盖: {bad}"


def dump(subset, name):
    old2new = {im["id"]: i for i, im in enumerate(subset)}
    anns = [a for a in d["annotations"] if a["image_id"] in old2new]
    images = [{**im, "id": old2new[im["id"]]} for im in subset]
    anns = [{**a, "id": i, "image_id": old2new[a["image_id"]]}
            for i, a in enumerate(anns)]
    json.dump({"images": images, "annotations": anns, "categories": d["categories"]},
              open(COCO / f"{name}.json", "w"))
    json.dump(texts, open(COCO / f"{name}_class_texts.json", "w"), ensure_ascii=False)
    return len(images), len(anns)


subset_val = sorted((im for im in imgs if im["id"] in val_ids), key=lambda x: x["id"])
subset_train = sorted((im for im in imgs if im["id"] not in val_ids), key=lambda x: x["id"])
ntr, atr = dump(subset_train, f"{SRC}_train")
nva, ava = dump(subset_val, f"{SRC}_val")

assert ntr + nva == len(imgs) and atr + ava == len(d["annotations"])
print(f"train {ntr} imgs / {atr} anns ; val {nva} imgs / {ava} anns (val {nva / len(imgs):.1%})")
print("val 分源:", dict(Counter(Path(im['file_name']).parts[0] for im in subset_val)))
vc = Counter(a["category_id"] for a in json.load(open(COCO / f"{SRC}_val.json"))["annotations"])
cov1 = sum(1 for v in vc.values() if v >= 1)
cov3 = sum(1 for v in vc.values() if v >= 3)
print(f"val 类覆盖: {cov1}/178 类 >=1 图, {cov3}/178 类 >=3 图; 未覆盖类 = 全语料仅 1 图的类（skis 例外被顺带覆盖）")
