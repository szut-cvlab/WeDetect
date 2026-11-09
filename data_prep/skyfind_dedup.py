#!/usr/bin/env python
"""SkyFind train 去重（按源内部），产出 COD 图清单 cod.json。

用法    python3 data_prep/skyfind_dedup.py [汉明阈值=6]
输入    /mnt/sda1/datasets/drone/SkyFind/train（剔 Visdrone_/DroneVehicle_）
输出    SkyFind/cod.json          {"root","images","stats","阈值"}
规则    1) 只在同一源内部比对
        2) dHash 汉明 ≤ 阈值视为同帧，LSH 分带找候选，贪心分簇
        3) 簇内取 Laplacian 方差最大的那张当代表
        4) 换代表后再按清晰度降序判一遍，保证最终保留集两两距离 > 阈值
"""
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageFile

D = "/mnt/sda1/datasets/drone/SkyFind/train"
OUT = "/mnt/sda1/datasets/drone/SkyFind/cod.json"
EXCL = ("Visdrone_", "DroneVehicle_")
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def features(path):
    """(dHash 64 位, 清晰度 = Laplacian 方差)。"""
    im = Image.open(path)
    im.draft("L", (512, 512))
    g = np.asarray(im.convert("L").resize((256, 256)), dtype=np.float32)
    a = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    h = int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).flat), 2)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return h, float(lap.var())


def popcount(x):
    return LUT[np.asarray(x, dtype=np.uint64).view(np.uint8).reshape(x.shape + (8,))].sum(-1)


def greedy(hashes, order, thr):
    """按给定顺序贪心：与已留者距离 ≤ thr 就丢。返回 (保留下标, 丢掉的下标→代表下标)。"""
    h = np.array([hashes[f] for f in order], dtype=np.uint64)
    bands, width = thr + 1, 64 // (thr + 1)
    keys = [(h >> np.uint64(b * width)) & ((np.uint64(1) << np.uint64(width)) - np.uint64(1))
            for b in range(bands)]
    keep, alias = [], {}
    buckets = [defaultdict(list) for _ in range(bands)]
    for i in range(len(order)):
        cand = set()
        for b in range(bands):
            cand |= set(buckets[b][int(keys[b][i])])
        if cand:
            idx = np.array(sorted(cand))
            d = popcount(h[i] ^ h[idx])
            j = int(d.argmin())
            if int(d[j]) <= thr:
                alias[i] = int(idx[j])
                continue
        keep.append(i)
        for b in range(bands):
            buckets[b][int(keys[b][i])].append(i)
    return keep, alias


def dedup_group(files, hashes, sharp, thr):
    # 第一遍：按文件名顺序分簇
    keep, alias = greedy(hashes, files, thr)
    clusters = defaultdict(list)
    for i in keep:
        clusters[i].append(i)
    for i, r in alias.items():
        clusters[r].append(i)

    # 簇内换成最清晰的那张
    rep_of = {}
    for members in clusters.values():
        best = max(members, key=lambda i: sharp[files[i]])
        for i in members:
            rep_of[i] = best

    # 第二遍：只在代表之间再判一次（清晰者优先），保证最终两两 > thr
    reps = sorted(set(rep_of.values()), key=lambda i: -sharp[files[i]])
    reps_order = [files[i] for i in reps]
    keep2, alias2 = greedy(hashes, reps_order, thr)
    final = {reps[i] for i in keep2}

    # 代表若在第二遍被更清晰的挤掉，映射到最终代表（alias2 的目标一定是 final 里的）
    to_final = {reps[i]: reps[j] for i, j in alias2.items()}
    out = {}
    for i, best in rep_of.items():
        if i in final:                       # 代表自己不算被删
            continue
        t = to_final.get(best, best)
        out[files[i]] = [files[t], bin(hashes[files[i]] ^ hashes[files[t]]).count("1")]
    return [files[i] for i in sorted(final)], out


def main():
    thr = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = [f for f in sorted(os.listdir(D)) if f.endswith(".jpg") and not f.startswith(EXCL)]
    print(f"待去重 {len(files)} 张（阈值 {thr}，按源内部）")
    with ThreadPoolExecutor(16) as ex:
        feats = list(ex.map(lambda f: features(f"{D}/{f}"), files))
    hashes = {f: v[0] for f, v in zip(files, feats)}
    sharp = {f: v[1] for f, v in zip(files, feats)}

    src = lambda f: f.split("_")[0] if f[0].isalpha() else "自爬"
    keep, dropped, stats = [], {}, {}
    for s in sorted({src(f) for f in files}):
        grp = [f for f in files if src(f) == s]
        k, d = dedup_group(grp, hashes, sharp, thr)
        keep += k
        dropped.update(d)
        stats[s] = {"输入": len(grp), "保留": len(k)}

    keep.sort()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"root": D, "images": keep, "stats": stats, "阈值": thr}, fh, ensure_ascii=False, indent=1)
    print(f"保留 {len(keep)} 张（去掉 {len(files) - len(keep)}）→ {OUT}")
    for s, v in sorted(stats.items(), key=lambda kv: -kv[1]["输入"]):
        print(f"  {s:16} {v['输入']:>6} → {v['保留']:>6}")


if __name__ == "__main__":
    main()
