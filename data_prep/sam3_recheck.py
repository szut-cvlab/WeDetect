#!/usr/bin/env python3
"""SAM 3 框提质全量：cod_uni_boxes_fodcoco.jsonl → 全部召回框 box-prompt 出 mask → mask2bbox 紧框替换，
垃圾框（碎框 / 框内无 mask 前景）丢弃；无响应但孤立（不在任何已认领 mask 内）的框保留原始框。
机制：整图 + 全部框一次前向（box prompt 撬动 200 个 DETR query 打分），288 分辨率上选实例，
每个 prompt 框按 IoU 认领最佳实例 mask（IoU<0.3 视为无响应），只对认领 mask 做 GPU 插值到原图。
输入  /mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco.jsonl
输出  /mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3.jsonl（同格式，紧框替换+丢框剔除，score 沿用原框）
断点  输出文件已含的 file 跳过；分片 SHARD/NUM_SHARDS（默认 0/1）；LIMIT=N 只跑前 N 张（冒烟）
抽检  SAMPLES 5 张另存工作区根目录 sam3_check_<stem>.jpg（红=原始框，绿=紧框，白=孤立保留，黄=丢弃）
运行  CUDA_VISIBLE_DEVICES=2 SHARD=0 NUM_SHARDS=2 .venv-sam3/bin/python data_prep/sam3_recheck.py
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm import tqdm
from PIL import Image, ImageDraw
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 池里有截断坏图，与 skyfind_dedup.py 同款处理

from transformers import Sam3Processor, Sam3Model

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / 'checkpoints/sam3'
COD = Path('/mnt/sda1/datasets/drone/SkyFind/cod_filtered.json')
JSONL_IN = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco.jsonl')
JSONL_OUT = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3.jsonl')
MATCH_IOU = 0.3    # prompt 框与实例框 IoU 下限，低于视为无响应
CAND_SCORE = 0.05  # 实例候选分数下限
SAMPLES = ['UVSD_3434.jpg', 'UVSD_1814.jpg', 'AUAIR_2172.jpg', 'Okutama-action_1102.jpg', '000811.jpg']


def mask2bbox(mask, W, H, box):
    """mask 裁剪到原始框内取紧框 (x1,y1,x2,y2)；框内无前景返回 None。
    只取框内部分：框外是泼出去的噪声，不参与紧框。"""
    bx1, by1, bx2, by2 = map(int, box)
    sub = mask.reshape(H, W)[by1:by2, bx1:bx2]
    ys, xs = sub.nonzero(as_tuple=True)
    if len(ys) == 0:
        return None
    return [bx1 + int(xs.min()), by1 + int(ys.min()),
            bx1 + int(xs.max()), by1 + int(ys.max())]


def load_records():
    """读输入 jsonl，按 SHARD/NUM_SHARDS 分片，跳过输出文件已有 file（断点续跑）"""
    shard = int(os.environ.get('SHARD', 0))
    n_shards = int(os.environ.get('NUM_SHARDS', 1))
    done = set()
    if JSONL_OUT.exists():
        for line in open(JSONL_OUT):
            try:
                done.add(json.loads(line)['file'])
            except Exception:
                pass
    recs = []
    for i, line in enumerate(open(JSONL_IN)):
        if i % n_shards != shard:
            continue
        r = json.loads(line)
        if r['file'] in done:
            continue
        recs.append(r)
    return recs


def process_image(model, proc, img, keep):
    """核心：前向 + 后处理。keep 为 xyxy 框列表。
    返回 (res, stats, matched)：res 与 keep 等长对齐，紧框或孤立保留为 xyxy、丢弃为 None；
    matched 为认领 mask（原图分辨率 bool，CPU），供抽检可视化。"""
    W, H = img.size
    with torch.no_grad():
        inputs = proc(images=img, input_boxes=[keep], return_tensors='pt').to('cuda')
        out = model(**inputs)

    # 只在 288 分辨率上选实例，只对认领 mask 做 GPU 插值（整包 post_process 会全插值到 4K，CPU 上几分钟）
    pred_masks = out.pred_masks[0]                  # (200, 288, 288) logits, cuda
    scores = out.pred_logits[0].sigmoid()
    if out.presence_logits is not None:
        scores = scores * out.presence_logits[0].sigmoid()
    pred_boxes = out.pred_boxes[0].cpu()            # (200, 4) xyxy 归一化
    cand = (scores > CAND_SCORE).nonzero(as_tuple=True)[0].cpu()
    cand_boxes = pred_boxes[cand] * torch.tensor([W, H, W, H], dtype=torch.float32)

    res = [None] * len(keep)
    stats = {'无响应碎框': 0, '无响应保留': 0, '空mask': 0, '紧框': 0}
    matched = []
    pb = torch.tensor(keep, dtype=torch.float32)
    if len(cand):
        best_iou, best_idx = box_iou(pb, cand_boxes).max(dim=1)
    else:
        best_iou = torch.zeros(len(keep))
        best_idx = torch.full((len(keep),), -1)

    unresp = []  # 无响应框 (i, box)
    for i, (x1, y1, x2, y2) in enumerate(keep):
        if best_iou[i] < MATCH_IOU:
            unresp.append((i, (x1, y1, x2, y2)))
            continue
        j = best_idx[i].item()
        logits = pred_masks[cand[j]][None, None]
        m = (F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)[0, 0] > 0).cpu()
        matched.append(m)
        bb = mask2bbox(m, W, H, (x1, y1, x2, y2))
        if bb:
            res[i] = bb
            stats['紧框'] += 1
        else:
            stats['空mask'] += 1
    # 无响应框二次判定：与任何已认领实例 mask 有重叠 → 碎框丢弃；完全孤立 → 保留原始框
    for i, (x1, y1, x2, y2) in unresp:
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cov = max((cm[iy1:iy2, ix1:ix2].float().mean().item() for cm in matched), default=0.0)
        if cov > 0:
            stats['无响应碎框'] += 1
        else:
            res[i] = [x1, y1, x2, y2]
            stats['无响应保留'] += 1
    return res, stats, matched


def draw_debug(img, keep, res, matched, name):
    """抽检可视化：mask 半透明贴原图 + 分色框，存工作区根目录"""
    refined = [b for b in res if b is not None]
    dropped = [keep[i] for i, b in enumerate(res) if b is None]
    kept_raw = [keep[i] for i, b in enumerate(res) if b == keep[i]]
    W, H = img.size
    rng = np.random.default_rng(0)
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    for m in matched:
        overlay[m.numpy()] = [*rng.integers(60, 255, 3), 110]
    img = Image.alpha_composite(img.convert('RGBA'), Image.fromarray(overlay)).convert('RGB')
    draw = ImageDraw.Draw(img)
    lw = max(2, min(W, H) // 1000)
    for x1, y1, x2, y2 in keep:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=lw)
    for x1, y1, x2, y2 in dropped:
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=lw + 2)
    for x1, y1, x2, y2 in refined:
        if [x1, y1, x2, y2] not in kept_raw:
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=lw * 2)
    for x1, y1, x2, y2 in kept_raw:
        draw.rectangle([x1, y1, x2, y2], outline='white', width=lw)
    draw.text((30, 30), f'{name}  raw={len(keep)} refined={len(refined) - len(kept_raw)} '
                        f'kept={len(kept_raw)} dropped={len(dropped)}', fill='white')
    img.resize((1600, int(1600 * H / W))).save(REPO / f'sam3_check_{Path(name).stem}.jpg', quality=85)


def main():
    cod = json.load(open(COD))
    root = Path(cod['root'])
    recs = load_records()
    limit = int(os.environ.get('LIMIT', 0))
    if limit:
        recs = recs[:limit]
    total = len(recs)
    print(f'本片 {total} 张', flush=True)
    if not total:
        return

    proc = Sam3Processor.from_pretrained(CKPT)
    model = Sam3Model.from_pretrained(CKPT).cuda().eval()

    tot = {'紧框': 0, '无响应保留': 0, '无响应碎框': 0, '空mask': 0}
    t_start = time.time()
    with open(JSONL_OUT, 'a') as fout:
        for k, r in enumerate(tqdm(recs, desc='sam3 refine', unit='img'), 1):
            if 'boxes' not in r:  # 缺图记录 {file, missing}，原样透传
                fout.write(json.dumps(r) + '\n')
                continue
            img = Image.open(root / r['file']).convert('RGB')
            boxes = r['boxes']
            keep = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes]
            if not keep:
                fout.write(json.dumps({'file': r['file'], 'width': r['width'],
                                       'height': r['height'], 'boxes': []}) + '\n')
                continue
            res, stats, matched = process_image(model, proc, img, keep)
            out_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1], src[4]]
                         for b, src in zip(res, boxes) if b is not None]
            rec = {'file': r['file'], 'width': r['width'], 'height': r['height'], 'boxes': out_boxes}
            fout.write(json.dumps(rec) + '\n')
            fout.flush()
            for key in tot:
                tot[key] += stats[key]
            if r['file'] in SAMPLES:
                draw_debug(img, keep, res, matched, r['file'])
    dt = time.time() - t_start
    print(f'完成 {total} 张，{dt / 3600:.2f}h，{dt / total:.1f}s/张，累计 {tot}', flush=True)


if __name__ == '__main__':
    main()
