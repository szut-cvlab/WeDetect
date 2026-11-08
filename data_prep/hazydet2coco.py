#!/usr/bin/env python
"""HazyDet -> COCO JSON（训练池口径）。

数据位置
    图像与标注  /mnt/sda1/datasets/drone/HazyDet
        {train,val,test}/{hazy_images,images}/*.jpg
        {train,val,test}/{split}_coco.json      标注指向雾图
        real_world/{train,test}/*.jpg + real_world/{train,test}_real.json

四类图的处理（实测数字见 notes/2026-11-07-uav-ovd-data.md §2.1.3）
    1 清晰图与雾图同名同框（5,907 场景）：只写清晰版。
      COCO 标注的是雾图，但同名的清晰图是同一场景、框完全共用，两版都写等于
      同一批实例计两次（234,354 框双计），所以按文件名配对后取清晰域。
    2 合成雾、无同名清晰图（5,093 张）：只有雾版一份，写 hazy_images。
    3 真实雾（600 张，与 RDDTS 是同一批，600/600 MD5 相同）：全部写。
    4 同帧双标注：同一帧被存了两份、两套标注互不包含程度的差异极小
      （例 000869/000870，像素均差 0.0，一份 37 框一份 19 框，并集只多 1）。
      判据按 §2.1「判重口径」：图像哈希完全相同且并集÷较大那张 ≤1.05 才算冗余，
      保留框数多的那份。注意仅"哈希相同"不构成删除理由——同机位不同时刻的框
      并集比值中位 1.48，那种两张都留。

类别
    官方 3 类 car / truck / bus，本池 category_id 从 0 起。
    类名不带下划线与连字符（这里本来就没有，沿用即可）。

产物
    coco/hazydet.json   整图，供 slice_coco.py 切块用
    类名表不在这里写，跟着入池的 hazydet_tiles.json 走

运行
    python3 data_prep/hazydet2coco.py
"""
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = "/mnt/sda1/datasets/drone/HazyDet"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
CLASSES = ["car", "truck", "bus"]
SPLITS = ["train", "val", "test"]
MERGE_UNION_RATIO = 1.05   # 并集÷较大那张 ≤ 此值才判为同帧冗余
IOU_TH = 0.5


def load_json(path):
    d = json.load(open(path))
    id2name = {i["id"]: i["file_name"] for i in d["images"]}
    id2wh = {i["id"]: (i["width"], i["height"]) for i in d["images"]}
    per_img = defaultdict(list)
    for a in d["annotations"]:
        per_img[a["image_id"]].append((a["category_id"], tuple(a["bbox"])))
    return id2name, id2wh, per_img, d["categories"]


def iou(x, y):
    ax2, ay2 = x[0] + x[2], x[1] + x[3]
    bx2, by2 = y[0] + y[2], y[1] + y[3]
    iw = max(0.0, min(ax2, bx2) - max(x[0], y[0]))
    ih = max(0.0, min(ay2, by2) - max(x[1], y[1]))
    it = iw * ih
    ar = x[2] * x[3] + y[2] * y[3] - it
    return it / ar if ar > 0 else 0.0


def union_ratio(A, B):
    """把两图的框并起来，返回 并集 ÷ 较大那张。"""
    if not A or not B:
        return 1.0
    extra = sum(1 for b in B if not any(iou(a[1], b[1]) >= IOU_TH for a in A))
    u = len(A) + extra
    return u / max(len(A), len(B))


def digest(path):
    from PIL import Image
    import imagehash
    try:
        with Image.open(path) as im:
            return path, str(imagehash.phash(im))
    except Exception:
        return path, None


# ---------- 收集候选 ----------
records = []      # (rel_path, width, height, [(cls_idx, bbox)...], 来源标签)
stat = Counter()

for sp in SPLITS:
    id2name, id2wh, per_img, cats = load_json(f"{ROOT}/{sp}/{sp}_coco.json")
    clear = set(os.listdir(f"{ROOT}/{sp}/images"))
    for iid, name in sorted(id2name.items()):
        boxes = per_img.get(iid, [])
        if not boxes:
            stat["空标注跳过"] += 1
            continue
        w, h = id2wh[iid]
        cls = Counter()
        conv = []
        for cid, bb in boxes:
            cn = next((c["name"] for c in cats if c["id"] == cid), None)
            if cn is None or cn not in CLASSES:
                stat["未知类别"] += 1
                continue
            x, y, bw, bh = bb
            if bw <= 0 or bh <= 0:
                stat["退化框"] += 1
                continue
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 0 or bh <= 0:
                stat["裁到图外"] += 1
                continue
            conv.append((CLASSES.index(cn), (x, y, bw, bh)))
            cls[cn] += 1
        if not conv:
            stat["清洗后变空"] += 1
            continue
        if name in clear:
            rel, tag = f"{sp}/images/{name}", "清晰图(有孪生)"
        else:
            rel, tag = f"{sp}/hazy_images/{name}", "合成雾(无孪生)"
        records.append((rel, w, h, conv, tag))
        stat[f"收 {tag}"] += 1

for sp in ["train", "test"]:
    id2name, id2wh, per_img, cats = load_json(f"{ROOT}/real_world/{sp}_real.json")
    for iid, name in sorted(id2name.items()):
        boxes = per_img.get(iid, [])
        if not boxes:
            stat["空标注跳过"] += 1
            continue
        w, h = id2wh[iid]
        conv = []
        for cid, bb in boxes:
            cn = next((c["name"] for c in cats if c["id"] == cid), None)
            if cn not in CLASSES:
                stat["未知类别"] += 1
                continue
            x, y, bw, bh = bb
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw <= 0 or bh <= 0:
                stat["退化框"] += 1
                continue
            conv.append((CLASSES.index(cn), (x, y, bw, bh)))
        if not conv:
            stat["清洗后变空"] += 1
            continue
        records.append((f"real_world/{sp}/{name}", w, h, conv, "真实雾"))
        stat["收 真实雾"] += 1

# ---------- 同帧冗余归并：图像哈希完全相同 且 并集比值 ≤1.05 ----------
paths = [f"{ROOT}/{r[0]}" for r in records]
with ProcessPoolExecutor(max_workers=min(32, (os.cpu_count() or 8))) as ex:
    hashes = dict(ex.map(digest, paths, chunksize=32))
by_hash = defaultdict(list)
for i, p in enumerate(paths):
    if hashes.get(p):
        by_hash[hashes[p]].append(i)

drop = set()
for hh, idxs in by_hash.items():
    if len(idxs) < 2:
        continue
    idxs = sorted(idxs, key=lambda i: -len(records[i][3]))   # 框多的排前，当代表
    keep = [idxs[0]]
    for i in idxs[1:]:
        merged = False
        for k in keep:
            if union_ratio(tuple(records[k][3]), tuple(records[i][3])) <= MERGE_UNION_RATIO:
                drop.add(i)
                merged = True
                break
        if not merged:
            keep.append(i)
stat["同帧冗余剔除"] = len(drop)

records = [r for i, r in enumerate(records) if i not in drop]

# ---------- 写 COCO ----------
images, anns = [], []
aid = 0
oor = 0
for rel, w, h, conv, tag in records:
    iid = len(images)
    images.append({"id": iid, "file_name": f"HazyDet/{rel}", "width": w, "height": h})
    for c, (x, y, bw, bh) in conv:
        if x < 0 or y < 0 or x + bw > w + 0.01 or y + bh > h + 0.01:
            oor += 1
        aid += 1
        anns.append({"id": aid, "image_id": iid, "category_id": c,
                     "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                     "area": round(bw * bh, 1), "iscrowd": 0})

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/hazydet.json", "w") as f:
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": i, "name": n, "supercategory": ""}
                              for i, n in enumerate(CLASSES)]}, f)

print(f"images={len(images)} boxes={len(anns)} 越界={oor}")
for k, v in sorted(stat.items()):
    print(f"  {k:14} {v}")
cnt = Counter(a["category_id"] for a in anns)
for i, n in enumerate(CLASSES):
    print(f"  {n:8} {cnt[i]}")
side = [min(a["bbox"][2], a["bbox"][3]) for a in anns]
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    c = sum(1 for s in side if lo <= s < hi)
    print(f"  edge [{lo},{hi}) = {c} ({c/len(side)*100:.1f}%)")
per = Counter(a["image_id"] for a in anns)
import statistics
print(f"  每图框数 中位 {statistics.median(per.values()):.0f} 均值 {len(anns)/len(images):.1f}")
# 路径校验：file_name 一律相对 /mnt/sda1/datasets/drone，逐条确认可达
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(DRONE_ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
