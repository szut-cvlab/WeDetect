#!/usr/bin/env python
"""DroneVehicle RGB 半边 -> COCO JSON。

数据位置
    图像  /mnt/sda1/datasets/drone/DroneVehicle/{train/trainimg,val/valimg,test/testimg}
    标注  同级的 {train/trainlabel,val/vallabel,test/testlabel}/*.xml
          CVAT 导出，每个 object 只带一种几何：
            polygon  八点 OBB，取 min/max 转水平框
            bndbox   xmin/ymin/xmax/ymax，本身已是水平框
          两者几乎不相交（混合文件里 bndbox 对 polygon 的最佳 IoU 全部 <0.02），
          所以两种都要收，不存在谁更全。
    红外  *imgr / *labelr 全部忽略，当前 RGB 模型不吃。

类别
    5 类：car、truck、bus、van、freight car（官方 classTitle 为 freight_car，按类名规则去下划线）。
    原始 name 有拼写错误，按 dataset.yaml 的别名表归一（truvk、feright car 等）。
    name 为 `*` 或没有几何的对象直接丢弃。

空标注
    785 个文件 0 目标（train 512）。这些图 ±3 帧邻域平均 15.7 个目标，是漏标不是没车，
    整批不写入 JSON——进了池子就是一张全是假负样本的图。等补标后再放进来。

不去重
    按 droneclip 的 64bit pHash + 全局贪心 t=10 复现过，删除率 60.4%，但被删图里 80.2%
    的框在保留图找不到任何 IoU >= 0.1 的目标，删的是不同场景，本池不去重。

产物
    /mnt/sda1/datasets/drone/coco/dronevehicle.json
    /mnt/sda1/datasets/drone/coco/dronevehicle_class_texts.json  [["car"], ...]

运行
    python3 data_prep/dronevehicle2coco.py
"""
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = "/mnt/sda1/datasets/drone/DroneVehicle"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
SPLITS = [("train", "train/trainimg", "train/trainlabel"),
          ("val", "val/valimg", "val/vallabel"),
          ("test", "test/testimg", "test/testlabel")]
# 类名规则：只用空格分隔。官方 classTitle freight_car 归一为 freight car，
# 与 ATR-UMOD 的 freight car 合并成同一个类；错拼由 ALIAS 收敛。
CLASSES = ["car", "truck", "bus", "van", "freight car"]
ALIAS = {"car": "car", "truck": "truck", "truvk": "truck", "bus": "bus", "van": "van",
         "feright car": "freight car", "feright_car": "freight car",
         "feright": "freight car", "freight_car": "freight car",
         "freight car": "freight car"}
NAME2ID = {n: i for i, n in enumerate(CLASSES)}
images, anns = [], []
cats = [{"id": i, "name": n, "supercategory": "vehicle"} for i, n in enumerate(CLASSES)]
aid = 0
stat = Counter()
edges = []

for sp, idir, ldir in SPLITS:
    img_dir, lab_dir = f"{ROOT}/{idir}", f"{ROOT}/{ldir}"
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".jpg"):
            continue
        stem = fn[:-4]
        xp = f"{lab_dir}/{stem}.xml"
        if not os.path.exists(xp):
            stat["no_xml"] += 1
            continue
        try:
            root = ET.parse(xp).getroot()
        except Exception:
            stat["xml_fail"] += 1
            continue
        size = root.find("size")
        w, h = int(size.findtext("width")), int(size.findtext("height"))
        objs = []
        for o in root.findall("object"):
            name = ALIAS.get((o.findtext("name") or "").strip().lower())
            pg, bb = o.find("polygon"), o.find("bndbox")
            if name is None:
                stat["bad_label"] += 1
                continue
            if pg is not None:
                xs = [float(pg.find(f"x{k}").text) for k in (1, 2, 3, 4)]
                ys = [float(pg.find(f"y{k}").text) for k in (1, 2, 3, 4)]
                box = (min(xs), min(ys), max(xs), max(ys))
                stat["from_polygon"] += 1
            elif bb is not None:
                box = (float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                       float(bb.findtext("xmax")), float(bb.findtext("ymax")))
                stat["from_bndbox"] += 1
            else:
                stat["no_geometry"] += 1
                continue
            x1, y1 = max(0.0, box[0]), max(0.0, box[1])
            x2, y2 = min(float(w), box[2]), min(float(h), box[3])
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                stat["degenerate"] += 1
                continue
            objs.append((NAME2ID[name], x1, y1, bw, bh))
        if not objs:
            stat["empty_skipped"] += 1
            continue
        iid = len(images)
        images.append({"id": iid, "file_name": f"DroneVehicle/{idir}/{fn}", "width": w, "height": h})
        for c, x1, y1, bw, bh in objs:
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                         "area": round(bw * bh, 1), "iscrowd": 0})
            edges.append(min(bw, bh))


os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/dronevehicle.json", "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": cats}, f)
with open(f"{OUT_DIR}/dronevehicle_class_texts.json", "w") as f:
    json.dump([[c["name"]] for c in cats], f, ensure_ascii=False)

print(f"images={len(images)} boxes={len(anns)}")
for k, v in sorted(stat.items()):
    print(f"  {k:14s} {v}")
cnt = Counter(a["category_id"] for a in anns)
for i, n in enumerate(CLASSES):
    print(f"  {n:12s} {cnt[i]}")
by_split = Counter(im["file_name"].split("/")[1] for im in images)
print("  分 split:", dict(by_split))
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    c = sum(1 for e in edges if lo <= e < hi)
    print(f"  edge [{lo},{hi}) = {c} ({c/len(edges)*100:.1f}%)")
# 路径校验：file_name 一律相对 /mnt/sda1/datasets/drone，逐条确认可达
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(DRONE_ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
