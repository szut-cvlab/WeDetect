#!/usr/bin/env python
"""VisDrone 离线指标：AP/AR 按目标大小分四档、per-class AP、IoU0.5@500 最优 F1。"""
import argparse
import pickle

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from build_coco import ZH

BINS = [("all", 0.0, 1e10), ("<16", 0.0, 16.0 ** 2), ("16-32", 16.0 ** 2, 32.0 ** 2),
        ("32-64", 32.0 ** 2, 64.0 ** 2), (">64", 64.0 ** 2, 1e10)]
N_POS = len(ZH)


class TqdmCOCOeval(COCOeval):
    """evaluate 逐 (类别, 图) 调 evaluateImg，包个进度条。"""

    def evaluate(self):
        n = len(self.params.catIds) * len(self.params.areaRng) * len(self.cocoGt.getImgIds())
        self._bar = tqdm(total=n, desc="evaluate(类×面积箱×图)", leave=False)
        super().evaluate()
        self._bar.close()

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        self._bar.update()
        return super().evaluateImg(imgId, catId, aRng, maxDet)


def evaluate(coco, dets, cat_ids, iou_thrs, max_dets):
    e = TqdmCOCOeval(coco, coco.loadRes(dets), "bbox")
    e.params.catIds = cat_ids
    e.params.iouThrs = iou_thrs
    e.params.maxDets = max_dets
    e.params.areaRng = [[lo, hi] for _, lo, hi in BINS]
    e.params.areaRngLbl = [n for n, _, _ in BINS]
    e.evaluate()
    e.accumulate()
    return e


def mean_ap(e, a, m):
    p = e.eval["precision"][:, :, :, a, m]
    return p[p > -1].mean() if (p > -1).any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="DumpDetResults 落盘的 pkl")
    ap.add_argument("--ann", default="data/VisDrone/annotations/visdrone_testdev_coco.json")
    args = ap.parse_args()

    coco = COCO(args.ann)
    samples = pickle.load(open(args.out, "rb"))
    dets = []
    for s in samples:
        pi = s["pred_instances"]
        for bbox, label, score in zip(pi["bboxes"], pi["labels"], pi["scores"]):
            x1, y1, x2, y2 = map(float, bbox)
            dets.append({"image_id": s["img_id"], "category_id": int(label) + 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    print(f"pkl: {len(samples)} imgs, {len(dets)} dets")

    e = evaluate(coco, dets, list(range(1, N_POS + 1)),
                 np.linspace(0.5, 0.95, 10, endpoint=True), [100, 300, 500])
    print("\nAP@[.5:.95]  " + "  ".join(f"{n:>6}" for n, _, _ in BINS))
    for m, tag in enumerate(["@100", "@500"]):
        mi = [0, 2][m]
        print(f"{tag:<12}" + "  ".join(f"{mean_ap(e, a, mi):6.3f}" for a in range(len(BINS))))

    print("\nAR@[.5:.95]  " + "  ".join(f"{n:>6}" for n, _, _ in BINS))
    for m, tag in enumerate(["@100", "@300", "@500"]):
        row = []
        for a in range(len(BINS)):
            r = e.eval["recall"][:, :, a, m]  # recall 是 [T,K,A,M]，无 R 维
            r = r[r > -1]
            row.append(f"{(r.mean() if r.size else float('nan')):6.3f}")
        print(f"{tag:<12}" + "  ".join(row))

    gt_num = [0] * N_POS
    for a in coco.anns.values():
        if a["category_id"] <= N_POS:
            gt_num[a["category_id"] - 1] += 1
    print("\nper-class AP@500 (GT 框数):")
    for k, name in enumerate(ZH):
        p = e.eval["precision"][:, :, k, 0, 2]
        p = p[p > -1]
        print(f"  {name:<6} {p.mean() if len(p) else float('nan'):6.3f}  ({gt_num[k]})")

    # IoU0.5@500 的 F1 直接取上面那次评测（IoU0.5 是第 0 档），不重复跑
    rec = e.params.recThrs
    p = e.eval["precision"][0, :, :, 0, 2]
    p = np.where(p > -1, p, np.nan)
    with np.errstate(invalid="ignore"):
        p = np.nanmean(p, axis=1)
        f1 = 2 * p * rec / np.maximum(p + rec, 1e-10)
        i = int(np.nanargmax(f1))
    print(f"\nIoU0.5 maxDets500: best-F1={f1[i]:.3f} @ P={p[i]:.3f} R={rec[i]:.3f}")


if __name__ == "__main__":
    main()
