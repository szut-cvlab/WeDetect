#!/usr/bin/env python
"""Uni 候选框召回率：AR@100/@300/@500 按 GT 目标大小分四档（<16/16-32/32-64/>64 像素）+ 候选数触顶检查。"""
import argparse
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import torch
from pycocotools.coco import COCO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "eval_recall"))
from recall import eval_recalls

BINS = [("all", 0.0, 1e10), ("<16", 0.0, 16.0 ** 2), ("16-32", 16.0 ** 2, 32.0 ** 2),
        ("32-64", 32.0 ** 2, 64.0 ** 2), (">64", 64.0 ** 2, 1e10)]
NUMS = [100, 300, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Uni pkl: [{'image_id','boxes'}]")
    ap.add_argument("--ann", default="data/VisDrone/annotations/visdrone_testdev_coco.json")
    args = ap.parse_args()

    coco = COCO(args.ann)
    gts = {}
    for img_id in coco.imgs:
        boxes = [a["bbox"] for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                 if not a.get("ignore", False) and not a["iscrowd"]]
        gts[img_id] = (np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], np.float32)
                       if boxes else np.zeros((0, 4), np.float32))
    preds = {i: None for i in coco.imgs}
    num_boxes = []
    for item in torch.load(args.pred, map_location="cpu", weights_only=False):
        preds[item["image_id"]] = np.asarray(item["boxes"], np.float32)
        num_boxes.append(len(preds[item["image_id"]]))
    missing = [i for i, v in preds.items() if v is None]
    assert not missing, f"pkl 缺 {len(missing)} 图"

    print(f"候选数/图: mean={np.mean(num_boxes):.1f} max={max(num_boxes)} "
          f"打满{max(num_boxes)}: {sum(n == max(num_boxes) for n in num_boxes)}/{len(num_boxes)}")

    iou_thrs = np.linspace(0.5, 0.95, 10, endpoint=True)
    print("\nAR@[.5:.95]  " + "  ".join(f"@{n:<5d}" for n in NUMS))
    for name, lo, hi in BINS:
        gt_list, pred_list = [], []
        for img_id, gt in gts.items():
            if len(gt):
                area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                gt = gt[(area > lo) & (area <= hi)]
                if len(gt) == 0:
                    gt = np.zeros((0, 4), np.float32)
            gt_list.append(gt)
            pred_list.append(preds[img_id])
        with redirect_stdout(io.StringIO()):
            r = eval_recalls(gt_list, pred_list, NUMS, iou_thrs)
        print(f"{name:<12}" + "  ".join(f"{r[i].mean():6.3f}" for i in range(len(NUMS))))


if __name__ == "__main__":
    main()
