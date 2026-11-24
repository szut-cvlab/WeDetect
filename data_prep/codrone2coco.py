#!/usr/bin/env python
"""CODrone 原始标注 -> 整图 COCO（未切块，供 slice_coco.py 用）。

数据位置
    /mnt/sda1/datasets/drone/CODrone/{train,val,test}/images/*.jpg   全部 3840×2160
    /mnt/sda1/datasets/drone/CODrone/{train,val,test}/annfile/*.txt   DOTA 风格：
        x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
        12 类加一个 ignored；difficulty 实测全 0
    labels/ 是同内容的 XML，不用它（与 annfile 等价，重复）

类别与命名
    池内类名一律空格分隔：官方 traffic-sign→traffic sign、traffic-light→traffic light，
    其余（car people motor truck boat bus bicycle tricycle ship bridge）原样小写。
    ignored 16,804 个写进 images[].ignore_regions，不进 annotations（与 VisDrone 同，
    训练代码不消费该字段）。

OBB -> 水平框
    四顶点取 min/max 得轴对齐外接矩形，再写 [x,y,w,h]。WeDetect 检测头只出水平框。

产物
    coco/codrone.json            整图，file_name 形如 CODrone/train/images/xxx.jpg（相对 drone/，与其余集一致）
    coco/codrone_class_texts.json  类名表
    切块版 codrone_tiles.json 由 slice_coco.py 生成，其 file_name 重写为相对 drone/ 的 tiles 路径

运行
    python3 data_prep/codrone2coco.py
"""
import json
import os
from collections import Counter

ROOT = "/mnt/sda1/datasets/drone/CODrone"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
SPLITS = ["train", "val", "test"]
# 统一词表 id（COD+FOD 合并词表，见 notes/2026-11-07 §2.3）
VOCAB = json.load(open(f"{OUT_DIR}/vocab_merged.json"))
NAME2UID = {n: i for i, n in enumerate(VOCAB)}
CLASSES = ["car", "people", "motorcycle", "truck", "traffic sign", "traffic light",
           "boat", "bus", "bicycle", "tricycle", "ship", "bridge"]
NAME2ID = {n: NAME2UID[n] for n in CLASSES}
RAW2NAME = {"motor": "motorcycle",
            "traffic-sign": "traffic sign", "traffic-light": "traffic light",
            "traffic_sign": "traffic sign", "traffic_light": "traffic light"}

images, anns = [], []
aid = 0
stat = Counter()
sides = []

for sp in SPLITS:
    adir, idir = f"{ROOT}/{sp}/annfile", f"{ROOT}/{sp}/images"
    for fn in sorted(os.listdir(adir)):
        if not fn.endswith(".txt"):
            continue
        jp = f"{idir}/{fn[:-4]}.jpg"
        if not os.path.exists(jp):
            stat["图像缺失"] += 1
            continue
        w, h = 3840, 2160
        objs, ign = [], []
        for line in open(f"{adir}/{fn}"):
            p = line.split()
            if len(p) != 10:
                stat["字段数不对"] += 1
                continue
            xy = list(map(float, p[:8]))
            x1, y1 = max(0.0, min(xy[0::2])), max(0.0, min(xy[1::2]))
            x2, y2 = min(float(w), max(xy[0::2])), min(float(h), max(xy[1::2]))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stat["退化框"] += 1
                continue
            if p[8] == "ignored":
                ign.append([round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)])
                stat["ignored"] += 1
                continue
            raw = p[8].strip()
            name = RAW2NAME.get(raw, raw)
            if name not in NAME2ID:
                stat[f"未知类别:{raw}"] += 1
                continue
            objs.append((NAME2ID[name], x1, y1, bw, bh))
        iid = len(images)
        images.append({"id": iid, "file_name": f"CODrone/{sp}/images/{fn[:-4]}.jpg",
                       "width": w, "height": h, "ignore_regions": ign})
        for c, x1, y1, bw, bh in objs:
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                         "area": round(bw * bh, 1), "iscrowd": 0})
            sides.append(min(bw, bh))
            stat["框"] += 1

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"images": images, "annotations": anns,
           "categories": [{"id": NAME2ID[n], "name": n, "supercategory": ""}
                          for n in CLASSES]},
          open(f"{OUT_DIR}/codrone.json", "w"))
json.dump([[n] for n in CLASSES], open(f"{OUT_DIR}/codrone_class_texts.json", "w"),
          ensure_ascii=False)

print(f"images={len(images)} boxes={len(anns)}")
for k, v in sorted(stat.items()):
    print(f"  {k:10} {v}")
cnt = Counter(a["category_id"] for a in anns)
for n in CLASSES:
    print(f"  {n:14} {cnt[NAME2ID[n]]}")
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    c = sum(1 for s in sides if lo <= s < hi)
    print(f"  edge [{lo},{hi}) = {c} ({c/len(sides)*100:.1f}%)")

# 路径校验：file_name 相对 /mnt/sda1/datasets/drone
missing = [im["file_name"] for im in images if not os.path.exists(f"{DRONE_ROOT}/{im['file_name']}")]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
