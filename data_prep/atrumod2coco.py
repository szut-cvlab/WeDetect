#!/usr/bin/env python
"""ATR-UMOD -> COCO JSON（训练池口径）。

数据位置
    /mnt/sda1/datasets/drone/ATR-UMOD/{train,test}/{images,labels}
    images  640×512，train 11,850 + test 1,503 = 13,353（与论文一致）
    labels  每图一个 XML：<size>、<UAVangle>/<UAVheight>/<weather>/<illumination>/<time>/<location>，
            每个 <object> 带 name、type、robndbox（cx/cy/w/h/angle）与 polygon（x1..x4,y1..y4）。

只取 RGB
    池子规则不收非 RGB 模态，红外半边已在本地删除，只剩一列 RGB 图（文件名 00001.jpg…），所以全部收。

类别（11 类车辆）
    源里是小写下划线写法，按池内规则处理：freight_car→freight car、tank_truck→tank truck（分隔符归一）；
    motorcycle→motor（与 VisDrone/CODrone 同概念，合并到多数派）；suv→SUV（缩写写法）。
    其余原样：car van bus truck trailer excavator crane。

OBB -> 水平框
    取 polygon 四顶点的 min/max 得轴对齐外接矩形，再写 [x,y,w,h]。WeDetect 检测头只出水平框。

判重
    按 §2.1 判重口径实测（抽 4,000 对相邻编号帧）：并集÷较大那张 中位 1.71，
    A→B 框匹配率中位 0.00 —— 重打包时文件名被重新编号，相邻编号不是同一场景的相邻帧，
    没有重复证据，全收。

空标注
    108 个 XML 无 object，跳过不入池（与其他集同口径）。

产物
    coco/atrumod.json                13,353 图档位；file_name 形如 ATR-UMOD/train/images/00001.jpg
    coco/atrumod_class_texts.json    11 类

运行
    python3 data_prep/atrumod2coco.py
"""
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = "/mnt/sda1/datasets/drone/ATR-UMOD"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
SPLITS = ["train", "test"]
CLASSES = ["car", "SUV", "van", "bus", "freight car", "truck",
           "motor", "tank truck", "trailer", "excavator", "crane"]
NAME2ID = {n: i for i, n in enumerate(CLASSES)}
# 分隔符归一：freight_car→freight car、tank_truck→tank truck
# 同概念合并：motorcycle→motor（VisDrone/CODrone 都叫 motor，多数派）
# 大小写：suv→SUV（缩写，论文写法；与 suv 的 embedding 余弦 0.987，无实质差异）
RAW2NAME = {"freight_car": "freight car", "tank_truck": "tank truck",
            "motorcycle": "motor", "suv": "SUV"}

images, anns = [], []
stat = Counter()
sides = []

for sp in SPLITS:
    ldir, idir = f"{ROOT}/{sp}/labels", f"{ROOT}/{sp}/images"
    for fn in sorted(os.listdir(ldir)):
        if not fn.endswith(".xml"):
            continue
        stem = fn[:-4]
        jp = f"{idir}/{stem}.jpg"
        if not os.path.exists(jp):
            stat["图像缺失"] += 1
            continue
        root = ET.parse(f"{ldir}/{fn}").getroot()
        w = float(root.findtext("size/width") or 640)
        h = float(root.findtext("size/height") or 512)
        conv = []
        for ob in root.findall("object"):
            raw = (ob.findtext("name") or "").strip()
            name = RAW2NAME.get(raw, raw)
            if name not in NAME2ID:
                stat[f"未知类别:{raw}"] += 1
                continue
            pg = ob.find("polygon")
            pts = []
            if pg is not None:
                for k in (1, 2, 3, 4):
                    xs, ys = pg.findtext(f"x{k}"), pg.findtext(f"y{k}")
                    if xs and ys:
                        pts.append((float(xs), float(ys)))
            if len(pts) < 3:
                stat["无几何"] += 1
                continue
            x1, y1 = max(0.0, min(p[0] for p in pts)), max(0.0, min(p[1] for p in pts))
            x2, y2 = min(w, max(p[0] for p in pts)), min(h, max(p[1] for p in pts))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stat["退化框"] += 1
                continue
            conv.append((NAME2ID[name], x1, y1, bw, bh))
        if not conv:
            stat["空标注跳过"] += 1
            continue
        iid = len(images)
        images.append({"id": iid, "file_name": f"ATR-UMOD/{sp}/images/{stem}.jpg",
                       "width": int(w), "height": int(h)})
        for c, x1, y1, bw, bh in conv:
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": c,
                         "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                         "area": round(bw * bh, 1), "iscrowd": 0})
            sides.append(min(bw, bh))
            stat["框"] += 1

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"images": images, "annotations": anns,
           "categories": [{"id": i, "name": n, "supercategory": ""}
                          for i, n in enumerate(CLASSES)]},
          open(f"{OUT_DIR}/atrumod.json", "w"))
json.dump([[n] for n in CLASSES], open(f"{OUT_DIR}/atrumod_class_texts.json", "w"),
          ensure_ascii=False)

print(f"images={len(images)} boxes={len(anns)}")
for k, v in sorted(stat.items()):
    print(f"  {k:12} {v}")
cnt = Counter(a["category_id"] for a in anns)
for i, n in enumerate(CLASSES):
    print(f"  {n:12} {cnt[i]}")
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    c = sum(1 for s in sides if lo <= s < hi)
    print(f"  edge [{lo},{hi}) = {c} ({c/len(sides)*100:.1f}%)")

missing = [im["file_name"] for im in images if not os.path.exists(f"{DRONE_ROOT}/{im['file_name']}")]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
