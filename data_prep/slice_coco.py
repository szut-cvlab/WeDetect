#!/usr/bin/env python
"""整图 COCO -> 1280 块 COCO。规则见 notes/2026-11-04-uav-ovd-plan.md §切图。

输入  coco/{dataset}.json（未切块，由各自的 *_2coco.py 产出）
输出  {dataset}/tiles/{split}/{stem}_x{i}_y{j}.jpg
      coco/{dataset}_tiles.json（file_name 相对 /mnt/sda1/datasets/drone）
      coco/{dataset}_class_texts.json

切法
    每条边独立判断：边长 <= 1600 整边保留（进 mosaic 缩放 >=0.8，可接受）；
    边长 > 1600 按 n = ceil(边长/1280)、步长 = (边长-1280)/(n-1) 均匀切。
    块尺寸：切的边恒 1280，留的边取原边长（1281-1600，不补黑边）。
    任一边 > 1600 的图切块落盘；两边都 <= 1600 的原图登记。

归属
    目标中心落在块内就写该块，框裁到块边界；重叠带里几块各写一份。
    裁后小于 4px 的丢弃。零目标块不写盘、不登记。

用法
    python3 data_prep/slice_coco.py --ann codrone.json --dataset CODrone
    python3 data_prep/slice_coco.py --ann hazydet.json --dataset HazyDet
"""
import argparse, json, math, os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image

ROOT = "/mnt/sda1/datasets/drone"
TILE, LONG_SIDE, MIN_SIDE = 1280, 1600, 4


def slug(name):
    """输出文件名用的小写名：去掉连字符、下划线、空格。"""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def edge_blocks(length):
    """一条边的 (起点, 尺寸) 列表：<=1600 整边一块，否则按 1280 均匀切。"""
    if length <= LONG_SIDE:
        return [(0, length)]
    n = math.ceil(length / TILE)
    step = (length - TILE) / (n - 1)
    return [(round(i * step), TILE) for i in range(n)]


def work(task):
    """处理一张图，返回 (image 记录列表, 碎片数, 零目标块数)。"""
    src, rel, w, h, boxes, ignores, cut = task

    if not cut:                                  # 两边都 <= 1600，原图登记
        if not boxes:
            return [], 0, 1
        return [(rel, w, h, boxes, ignores)], 0, 0

    img = Image.open(src)
    if img.size != (w, h):
        return [], 0, 1                          # 尺寸与标注不符，跳过
    out, drop, empty = [], 0, 0
    for j, (ty, th) in enumerate(edge_blocks(h)):
        for i, (tx, tw) in enumerate(edge_blocks(w)):
            kept = []
            for cid, x, y, bw, bh in boxes:
                if not (tx <= x + bw / 2 < tx + tw and ty <= y + bh / 2 < ty + th):
                    continue                     # 中心不在本块
                x1, y1 = max(x, tx), max(y, ty)
                x2, y2 = min(x + bw, tx + tw), min(y + bh, ty + th)
                if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                    drop += 1
                    continue
                kept.append([cid, round(x1 - tx, 2), round(y1 - ty, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)])
            if not kept:                         # 零目标块：不写盘
                empty += 1
                continue
            own = [[round(max(x, tx) - tx, 2), round(max(y, ty) - ty, 2),
                    round(min(x + bw, tx + tw) - max(x, tx), 2),
                    round(min(y + bh, ty + th) - max(y, ty), 2)]
                   for x, y, bw, bh in ignores
                   if tx <= x + bw / 2 < tx + tw and ty <= y + bh / 2 < ty + th]
            p = rel.replace("{i}", str(i)).replace("{j}", str(j))
            os.makedirs(os.path.dirname(f"{ROOT}/{p}"), exist_ok=True)
            img.crop((tx, ty, tx + tw, ty + th)).save(f"{ROOT}/{p}", "JPEG", quality=92)
            out.append((p, tw, th, kept, own))
    return out, drop, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="未切块 COCO，相对 coco/ 或绝对路径")
    ap.add_argument("--dataset", required=True, help="数据集目录名，如 CODrone")
    a = ap.parse_args()

    d = json.load(open(a.ann if os.path.isabs(a.ann) else f"{ROOT}/coco/{a.ann}"))
    by_img = {}
    for an in d["annotations"]:
        by_img.setdefault(an["image_id"], []).append((an["category_id"], *an["bbox"]))

    tasks = []
    pref = a.dataset + "/"
    for im in d["images"]:
        fn = im["file_name"][len(pref):] if im["file_name"].startswith(pref) else im["file_name"]
        cut = max(im["width"], im["height"]) > LONG_SIDE
        rel = (f"{a.dataset}/tiles/{fn.split('/')[0]}/"
               f"{os.path.basename(fn)[:-4]}_x{{i}}_y{{j}}.jpg" if cut else f"{a.dataset}/{fn}")
        tasks.append((f"{ROOT}/{a.dataset}/{fn}", rel, im["width"], im["height"],
                      by_img.get(im["id"], []), im.get("ignore_regions", []), cut))

    records, drop, empty = [], 0, 0
    with ProcessPoolExecutor(max_workers=32) as ex:
        for recs, dr, emp in tqdm(ex.map(work, tasks, chunksize=8),
                                  total=len(tasks), desc="切块", unit="img"):
            records += recs
            drop += dr
            empty += emp

    images, anns = [], []
    for rel, w, h, kept, own in records:
        iid = len(images)
        images.append({"id": iid, "file_name": rel, "width": w, "height": h,
                       "ignore_regions": own})
        for cid, x, y, bw, bh in kept:
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cid,
                         "bbox": [x, y, bw, bh], "area": round(bw * bh, 1), "iscrowd": 0})

    out_stem = os.path.splitext(os.path.basename(a.ann))[0]
    json.dump({"images": images, "annotations": anns, "categories": d["categories"]},
              open(f"{ROOT}/coco/{out_stem}_tiles.json", "w"))
    json.dump([[c["name"]] for c in d["categories"]],
              open(f"{ROOT}/coco/{slug(a.dataset)}_class_texts.json", "w"), ensure_ascii=False)

    print(f"出图 {len(images)}  框 {len(anns)}  零目标块丢弃 {empty}  碎片 {drop}")
    miss = [i["file_name"] for i in images if not os.path.exists(f"{ROOT}/{i['file_name']}")]
    if miss:
        print(f"警告：{len(miss)} 个 file_name 不存在，例如 {miss[:2]}")


if __name__ == "__main__":
    main()
