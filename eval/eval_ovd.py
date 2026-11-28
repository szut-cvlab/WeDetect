#!/usr/bin/env python
"""OVD 离线评测：mmdet DumpDetResults pkl -> 总 AP + 四档分档 AP；--ignore 启用 VisDrone ignore 处理。

label -> category_id 按 ann json 的 categories 顺序映射（评测 config 的 metainfo classes 同序）。

--ignore（VisDrone test-dev）：类 0/11 区域（json 元数据 ignore_regions/others_regions）在
评测内存中对每个真实类别注入 iscrowd=1 伪 GT——pycocotools 对 crowd 用 IoF 匹配，
未匹配 GT 且与其重叠超阈值的检出标记 dtIgnore 不计 FP（VisDrone 官方 devkit 语义）。
只在内存注入，不落盘、不污染 GT json 的 per-category 统计。
已知偏差：pycocotools 对单个检出取 IoF/IoU 最大的 gt，真 GT 与 ignore 区域同时高度重叠时
可能与官方 devkit 的"先匹配真 GT"顺序有差异，实际影响可忽略。
"""
import argparse
import json
import pickle

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

BINS = [("all", 0.0, 1e10), ("<16", 0.0, 16.0 ** 2), ("16-32", 16.0 ** 2, 32.0 ** 2),
        ("32-64", 32.0 ** 2, 64.0 ** 2), (">64", 64.0 ** 2, 1e10)]
IOU_THRS = np.linspace(0.5, 0.95, 10, endpoint=True)
MAX_DETS = [100, 300, 500]


class TqdmCOCOeval(COCOeval):
    def evaluate(self):
        n = len(self.params.catIds) * len(self.params.areaRng) * len(self.cocoGt.getImgIds())
        self._bar = tqdm(total=n, desc="evaluate", leave=False)
        super().evaluate()
        self._bar.close()

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        self._bar.update()
        return super().evaluateImg(imgId, catId, aRng, maxDet)


def load_dets(pkl_path, ann):
    """pkl -> COCO det list；label 按 categories 顺序映射回 category_id。"""
    cat_ids = [c["id"] for c in ann["categories"]]
    samples = pickle.load(open(pkl_path, "rb"))
    dets = []
    for s in samples:
        pi = s["pred_instances"]
        for bbox, label, score in zip(pi["bboxes"], pi["labels"], pi["scores"]):
            x1, y1, x2, y2 = map(float, bbox)
            dets.append({"image_id": int(s["img_id"]),
                         "category_id": int(cat_ids[int(label)]),
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "score": float(score)})
    return dets


def extend_with_ignore(coco):
    """内存注入：每图每个真实类别 × 每个类 0/11 区域一条 iscrowd=1 伪 GT。"""
    next_id = max(coco.anns) + 1 if coco.anns else 1
    n_inj = 0
    for img_id, im in coco.imgs.items():
        regions = list(im.get("ignore_regions", [])) + list(im.get("others_regions", []))
        for x, y, w, h in regions:
            for cat_id in coco.cats:
                coco.anns[next_id] = {
                    "id": next_id, "image_id": img_id, "category_id": cat_id,
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 1}
                coco.imgToAnns[img_id].append(coco.anns[next_id])
                next_id += 1
                n_inj += 1
    return n_inj


def run_eval(coco, dets, tag):
    e = TqdmCOCOeval(coco, coco.loadRes(dets), "bbox")
    e.params.catIds = sorted(coco.cats)
    e.params.iouThrs = IOU_THRS
    e.params.maxDets = MAX_DETS
    e.params.areaRng = [[lo, hi] for _, lo, hi in BINS]
    e.params.areaRngLbl = [n for n, _, _ in BINS]
    e.evaluate()
    e.accumulate()

    def ap(a, m):
        p = e.eval["precision"][:, :, :, a, m]
        return p[p > -1].mean() if (p > -1).any() else float("nan")

    def ap_at(t, a, m):
        p = e.eval["precision"][t, :, :, a, m]
        return p[p > -1].mean() if (p > -1).any() else float("nan")

    def ar(t, a, m):
        r = e.eval["recall"][t, :, a, m]
        return r[r > -1].mean() if (r > -1).any() else float("nan")

    def ar_all(a, m):
        r = e.eval["recall"][:, :, a, m]
        return r[r > -1].mean() if (r > -1).any() else float("nan")

    print(f"\n[{tag}]  " + "  ".join(f"{n:>6}" for n, _, _ in BINS)
          + "   (maxDets=500)")
    print(f"{'AP@.5:.95':<12}" + "  ".join(f"{ap(a, 2):6.3f}" for a in range(len(BINS))))
    print(f"{'AP@.5':<12}" + "  ".join(f"{ap_at(0, a, 2):6.3f}" for a in range(len(BINS))))
    print(f"{'AR@.5:.95':<12}" + "  ".join(f"{ar_all(a, 2):6.3f}" for a in range(len(BINS)))
          + "   (maxDets=500)")
    print(f"{'AR@.5':<12}" + "  ".join(f"{ar(0, a, 2):6.3f}" for a in range(len(BINS)))
          + "   (maxDets=500)")
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="mmdet DumpDetResults 的 pkl")
    ap.add_argument("--ann", required=True, help="考卷 COCO json（统一词表 id）")
    ap.add_argument("--ignore", action="store_true",
                    help="启用 VisDrone ignore_regions/others_regions 处理（仅 VisDrone json 支持）")
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    has_regions = any(im.get("ignore_regions") or im.get("others_regions")
                      for im in ann["images"])
    if args.ignore and not has_regions:
        raise SystemExit("--ignore 但 json 无 ignore_regions/others_regions 元数据")

    dets = load_dets(args.pkl, ann)
    print(f"pkl: {len(dets)} dets")

    coco = COCO(args.ann)
    run_eval(coco, dets, "standard")

    if args.ignore:
        n_inj = extend_with_ignore(coco)
        print(f"\n注入 ignore 伪 crowd ann {n_inj} 条")
        run_eval(coco, dets, "ignore-aware")


if __name__ == "__main__":
    main()
