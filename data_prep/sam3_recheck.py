#!/usr/bin/env python3
"""SAM 3 框提质抽检：fodcoco 召回框 → box-prompt 出 mask → mask2bbox 紧框替换，
垃圾框（无实例响应 / mask 糊图 / 贴边 / 空掩码）丢弃。红=原始框，绿=提质后，黄=丢弃。
用法：box prompt 撬动 200 个 query 打分，post_process 出全部实例，每个 prompt 框按 IoU 认领
最佳实例的 mask（IoU<0.3 视为无响应丢弃）。
运行  .venv-sam3/bin/python data_prep/sam3_recheck.py
输出  工作区根目录 sam3_check_<file>.jpg + stdout 统计
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import torch
from torchvision.ops import box_iou
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / 'checkpoints/sam3'
COD = Path('/mnt/sda1/datasets/drone/SkyFind/cod.json')
JSONL = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco.jsonl')
MIN_PX = 16      # 以下不过 SAM，直接全人工
MATCH_IOU = 0.3  # prompt 框与实例框 IoU 下限，低于视为无响应
FILL_THRE = 0.9  # mask 面积占图比例上限，超过视为糊图丢框
EDGE_PX = 4      # mask 距图边小于该值视为贴边丢框
SAMPLES = ['UVSD_3434.jpg', 'UVSD_1814.jpg', 'AUAIR_2172.jpg', 'Okutama-action_1102.jpg', '000000.jpg']


def mask2bbox(mask, W, H):
    """mask → 紧框 (x1,y1,x2,y2)，垃圾 mask 返回 None"""
    mask = mask.reshape(H, W)
    ys, xs = mask.nonzero(as_tuple=True)
    if len(ys) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    if (x2 - x1 + 1) * (y2 - y1 + 1) / (W * H) > FILL_THRE:
        return None  # 糊图
    if x1 < EDGE_PX or y1 < EDGE_PX or W - 1 - x2 < EDGE_PX or H - 1 - y2 < EDGE_PX:
        return None  # 贴边
    return [int(x1), int(y1), int(x2), int(y2)]


def main():
    cod = json.load(open(COD))
    root = Path(cod['root'])
    boxes_by_file = {}
    for line in open(JSONL):
        r = json.loads(line)
        if r['file'] in SAMPLES:
            boxes_by_file[r['file']] = r['boxes']

    proc = Sam3Processor.from_pretrained(CKPT)
    model = Sam3Model.from_pretrained(CKPT).cuda().eval()

    for name in SAMPLES:
        boxes = boxes_by_file.get(name)
        if boxes is None:
            print(f'{name}: 无召回框，跳过')
            continue
        keep = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes if b[2] >= MIN_PX and b[3] >= MIN_PX]
        img = Image.open(root / name).convert('RGB')
        W, H = img.size

        with torch.no_grad():
            inputs = proc(images=img, input_boxes=[keep], return_tensors='pt').to('cuda')
            out = model(**inputs)
        for k in ('pred_masks', 'pred_logits', 'pred_boxes', 'presence_logits'):
            if getattr(out, k, None) is not None:
                setattr(out, k, getattr(out, k).cpu())  # 4K mask 还原放 CPU，200 query × 原图尺寸 GPU 装不下
        res = proc.post_process_instance_segmentation(
            out, threshold=0.05, target_sizes=[[H, W]])[0]
        inst_boxes, inst_masks = res['boxes'], res['masks']  # (N,4) xyxy 像素 / (N,H,W) bool

        n_resp = 0
        refined, dropped = [], []
        if len(inst_boxes):
            pb = torch.tensor(keep, device=inst_boxes.device)
            iou = box_iou(pb, inst_boxes)  # (M, N)
            best_iou, best_idx = iou.max(dim=1)
        for i, (x1, y1, x2, y2) in enumerate(keep):
            if len(inst_boxes) == 0 or best_iou[i] < MATCH_IOU:
                dropped.append([x1, y1, x2, y2])
                continue
            n_resp += 1
            bb = mask2bbox(inst_masks[best_idx[i]].squeeze(), W, H)  # (1,H,W)→(H,W)
            if bb:
                refined.append(bb)
            else:
                dropped.append([x1, y1, x2, y2])

        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in dropped:
            draw.rectangle([x1, y1, x2, y2], outline='yellow', width=4)
        for x1, y1, x2, y2 in keep:
            draw.rectangle([x1, y1, x2, y2], outline='red', width=4)
        for x1, y1, x2, y2 in refined:
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
        draw.text((30, 30), f'{name}  raw={len(keep)} refined={len(refined)} dropped={len(dropped)}', fill='white')
        img.resize((1600, int(1600 * H / W))).save(REPO / f'sam3_check_{Path(name).stem}.jpg', quality=85)
        print(f'{name}: 原始 {len(keep)} → 响应 {n_resp} → 紧框 {len(refined)}，丢 {len(dropped)}')


if __name__ == '__main__':
    main()
