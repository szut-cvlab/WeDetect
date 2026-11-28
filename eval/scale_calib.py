#!/usr/bin/env python
"""打分侧免训练尺度校准（method §3.3）：ovd_val 上按边长分 4 桶拟合温度 σ(z/T_b)，迁移评到考卷。

拟合：检出 TP 标签 = IoU>=0.5 贪心匹配（逐图逐类，分数降序）；
每桶最小化 BCE(sigmoid(z/T_b), y) 求 T_b，z = logit(score)。
用途：拿"尺度偏差存在 + 轻量校准修不完"两句话证据，不占方法主线。

用法：
  python eval/scale_calib.py --fit-pkl work_dirs/baseline_tiny_ovdval.pkl \
      --fit-ann .../ovd_val.json --target .../visdrone_testdev.json=work_dirs/baseline_tiny_visdrone.pkl[:ignore] ...
"""
import argparse
import json
import math

import numpy as np
from scipy.optimize import minimize_scalar

from eval_ovd import BINS, COCO, TqdmCOCOeval, extend_with_ignore, load_dets

EDGE_BINS = BINS[1:]  # <16 / 16-32 / 32-64 / >64


def bucket_of(area):
    for b, (_, lo, hi) in enumerate(EDGE_BINS):
        if lo <= area < hi:
            return b
    return len(EDGE_BINS) - 1


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def tp_flags(dets, coco):
    """逐图逐类贪心匹配，返回每条检出是否 TP。"""
    gt_by = {}
    for a in coco.dataset["annotations"]:
        if a.get("iscrowd"):
            continue
        gt_by.setdefault((a["image_id"], a["category_id"]), []).append(a)
    order = {}
    for i, d in enumerate(dets):
        order.setdefault((d["image_id"], d["category_id"]), []).append(i)
    flags = np.zeros(len(dets), dtype=bool)
    for key, idxs in order.items():
        anns = gt_by.get(key, [])
        boxes = np.array([a["bbox"] for a in anns]).reshape(-1, 4)
        used = np.zeros(len(anns), dtype=bool)
        for i in sorted(idxs, key=lambda j: -dets[j]["score"]):
            if not len(anns):
                break
            x1, y1, w, h = dets[i]["bbox"]
            ix1 = np.maximum(boxes[:, 0], x1)
            iy1 = np.maximum(boxes[:, 1], y1)
            ix2 = np.minimum(boxes[:, 0] + boxes[:, 2], x1 + w)
            iy2 = np.minimum(boxes[:, 1] + boxes[:, 3], y1 + h)
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            union = boxes[:, 2] * boxes[:, 3] + w * h - inter
            iou = np.where(union > 0, inter / union, 0)
            iou[used] = 0
            g = int(iou.argmax())
            if iou[g] >= 0.5:
                used[g] = True
                flags[i] = True
    return flags


def fit_temperatures(dets, coco):
    """返回 4 个桶的 T_b（score' = sigmoid(logit / T)）。"""
    flags = tp_flags(dets, coco)
    zs = np.array([logit(d["score"]) for d in dets])
    bs = np.array([bucket_of(d["bbox"][2] * d["bbox"][3]) for d in dets])
    temps, info = [], []
    for b in range(len(EDGE_BINS)):
        m = bs == b
        z, y = zs[m], flags[m].astype(float)
        if m.sum() < 50 or y.mean() in (0.0, 1.0):
            temps.append(1.0)
            info.append((EDGE_BINS[b][0], int(m.sum()), float(y.mean()) if m.sum() else 0, 1.0, "n/a"))
            continue

        def bce(log_t):
            s = 1.0 / (1.0 + np.exp(-z / math.exp(log_t)))
            s = np.clip(s, 1e-7, 1 - 1e-7)
            return -(y * np.log(s) + (1 - y) * np.log(1 - s)).mean()

        r = minimize_scalar(bce, bounds=(-3, 3), method="bounded")
        T = float(math.exp(r.x))
        temps.append(T)
        s = 1.0 / (1.0 + np.exp(-z))
        info.append((EDGE_BINS[b][0], int(m.sum()), float(y.mean()), round(T, 4),
                     f"TP均分 {s[y > 0].mean():.3f} FP均分 {s[y < 1].mean():.3f} "
                     f"BCE {bce(0.0):.4f}->{r.fun:.4f}"))
    return temps, info


def rescore(dets, temps):
    out = []
    for d in dets:
        b = bucket_of(d["bbox"][2] * d["bbox"][3])
        d2 = dict(d)
        d2["score"] = 1.0 / (1.0 + math.exp(-logit(d["score"]) / temps[b]))
        out.append(d2)
    return out


def ap_summary(ann_path, dets, ignore=False):
    """返回 (AP@.5:.95 五档, AP@.5 五档)，maxDets=500。"""
    coco = COCO(ann_path)
    if ignore:
        extend_with_ignore(coco)
    e = TqdmCOCOeval(coco, coco.loadRes(dets), "bbox")
    e.params.maxDets = [100, 300, 500]
    e.params.areaRng = [[lo, hi] for _, lo, hi in BINS]
    e.params.areaRngLbl = [n for n, _, _ in BINS]
    e.evaluate()
    e.accumulate()
    p = e.eval["precision"]

    def ap(t, a):
        x = p[t, :, :, a, 2]
        return float(x[x > -1].mean()) if (x > -1).any() else float("nan")

    ap50 = [ap(0, a) for a in range(len(BINS))]
    apm = []
    for a in range(len(BINS)):
        x = p[:, :, :, a, 2]
        apm.append(float(x[x > -1].mean()) if (x > -1).any() else float("nan"))
    return apm, ap50


def fmt(v):
    return "  ".join(f"{x * 100:5.1f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-pkl", required=True)
    ap.add_argument("--fit-ann", required=True)
    ap.add_argument("--target", action="append", required=True,
                    help="ann.json=pkl[=ignore]，可多次")
    ap.add_argument("--out", default="work_dirs/scale_calib_T.json")
    args = ap.parse_args()

    fit_ann = json.load(open(args.fit_ann))
    dets = load_dets(args.fit_pkl, fit_ann)
    coco = COCO(args.fit_ann)
    temps, info = fit_temperatures(dets, coco)
    print(f"拟合集 {args.fit_pkl}  检出 {len(dets)}")
    print("桶        检出数   TP率    T_b     分数与拟合")
    for row in info:
        print(f"{row[0]:<8} {row[1]:>8}  {row[2]:.3f}  {row[3]:>7}  {row[4]}")
    json.dump({"edges": [n for n, _, _ in EDGE_BINS], "temps": temps,
               "fit_on": args.fit_pkl}, open(args.out, "w"))

    for spec in args.target:
        parts = spec.split("=")
        ann_p, pkl_p = parts[0], parts[1]
        ignore = len(parts) > 2 and parts[2] == "ignore"
        tann = json.load(open(ann_p))
        tdets = load_dets(pkl_p, tann)
        rdets = rescore(tdets, temps)
        tag = "ignore-aware" if ignore else "standard"
        (apm_b, ap50_b) = ap_summary(ann_p, tdets, ignore)
        (apm_a, ap50_a) = ap_summary(ann_p, rdets, ignore)
        print(f"\n迁移 -> {pkl_p}  [{tag}]")
        print("        " + "  ".join(f"{n:>5}" for n, _, _ in BINS))
        for name, b, af in [("AP@.5:.95", apm_b, apm_a), ("AP@.5", ap50_b, ap50_a)]:
            print(f"{name:<10}before " + fmt(b))
            print(f"{'':<10}after  " + fmt(af))
            print(f"{'':<10}dAP    {100 * (af[0] - b[0]):+.1f} pt")


if __name__ == "__main__":
    main()
