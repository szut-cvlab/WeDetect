#!/usr/bin/env python3
"""COD 图 Large-Uni 全召回：长边 >1280 内存切 1280×1280 tile，框映射回原图坐标，逐图追加 jsonl。

用法  python cod_uni_recall.py            # 双卡分片全量召回
      python cod_uni_recall.py viz        # 每数据源抽一张画召回框（<源>_sample.jpg）
单图测试：
    from cod_uni_recall import load_model, infer_image
    model = load_model()
    W, H, boxes = infer_image(model, Image.open('xxx.jpg'))
"""
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from generate_proposal import SimpleYOLOWorldDetector  # noqa: E402

TILE = 1280
SCORE_THRE = 0.1     # 并集后最终过滤
NUM_PROPOSALS = 500  # 每 tile top-K
CKPT = REPO / 'checkpoints/WeDetect/wedetect_large_uni_fodcoco.pth'
COD_JSON = '/mnt/sda1/datasets/drone/SkyFind/cod.json'
OUT_JSONL = '/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco.jsonl'
SHARD = int(os.environ.get('SHARD', '0'))      # 双卡跑：0 和 1，N_SHARDS=2
N_SHARDS = int(os.environ.get('N_SHARDS', '2'))


def load_model(ckpt_path=CKPT):
    model = SimpleYOLOWorldDetector(
        backbone_size='large', prompt_dim=768, num_prompts=256,
        num_proposals=NUM_PROPOSALS)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # 本机训练产物，含 mmengine 对象
    if 'state_dict' in ckpt:  # mmengine 训练产物（epoch_N.pth），键名与 HF pth 同族
        ckpt = ckpt['state_dict']
    # 键名重映射照抄 generate_proposal.py __main__（:1236-1254）
    for key in list(ckpt.keys()):
        if 'backbone' in key:
            ckpt[key.replace('backbone.image_model.model.', 'backbone.')] = ckpt.pop(key)
    for key in list(ckpt.keys()):
        if 'bbox_head' in key:
            new_key = key.replace('bbox_head.head_module.', 'bbox_head.')
            for a, b in (('0.2.', '0.6.'), ('1.2.', '1.6.'), ('2.2.', '2.6.'),
                         ('1.bn', '4'), ('1.conv', '3'), ('0.bn', '1'), ('0.conv', '0')):
                new_key = new_key.replace(a, b)
            ckpt[new_key] = ckpt.pop(key)
    msg = model.load_state_dict(ckpt, strict=False)
    assert not msg.missing_keys, f'权重缺失: {msg.missing_keys}'
    model.cuda().eval()
    return model


def tile_starts(edge):
    if edge <= TILE:
        return [0]
    n = math.ceil(edge / TILE)
    stride = (edge - TILE) / (n - 1)
    starts = sorted({round(i * stride) for i in range(n)})
    starts[-1] = edge - TILE
    return starts


@torch.no_grad()
def infer_image(model, img):
    """一张 PIL 图 → (W, H, boxes)，boxes=[[x, y, w, h, score], ...] 原图坐标。

    后处理顺序：每 tile NMS 0.7 → 按 score 取 top NUM_PROPOSALS 截断 →
    跨 tile 并集 → SCORE_THRE 过滤 → NMS 0.5（tile 内 0.7 是源码自带的候选筛选）。
    """
    W, H = img.size
    if W <= TILE and H <= TILE:
        tiles = [(0, 0, img)]
    else:
        tiles = []
        for y0 in tile_starts(H):
            for x0 in tile_starts(W):
                tiles.append((x0, y0,
                              img.crop((x0, y0, min(x0 + TILE, W), min(y0 + TILE, H)))))

    def run_chunk(chunk):
        results = model([t[2] for t in chunk])
        out = []
        for (x0, y0, _), r in zip(chunk, results):
            b = r['bboxes'].float().cpu()
            s = r['scores'].float().cpu()
            if len(b):
                b = b + b.new_tensor([x0, y0, x0, y0])
            out.append((b, s))
        return out

    chunks = []
    for i in range(0, len(tiles), 12):  # 4080 16GB 实测：6 tile 才 6.7GB，12 tile 约 12GB 带余量
        chunks += run_chunk(tiles[i:i + 12])

    boxes = torch.cat([b for b, _ in chunks])
    scores = torch.cat([s for _, s in chunks])
    keep = scores > SCORE_THRE
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        keep = torchvision.ops.nms(boxes, scores, 0.5)  # 跨 tile 去重（0.7 留太多近重复框）
        boxes, scores = boxes[keep], scores[keep]
    out = [[round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1), round(float(sc), 4)]
           for (x1, y1, x2, y2), sc in zip(boxes.tolist(), scores.tolist())]
    return W, H, out


def viz(boxes_jsonl=None):
    """每个数据源抽一张图画召回框，存仓库根 <源>_sample.jpg"""
    boxes_jsonl = boxes_jsonl or OUT_JSONL
    cod = json.load(open(COD_JSON))
    root = Path(cod['root'])
    boxes_by_file = {}
    for line in open(boxes_jsonl):
        r = json.loads(line)
        if 'boxes' in r:
            boxes_by_file[r['file']] = r['boxes']

    groups = defaultdict(list)
    for f in cod['images']:
        if f in boxes_by_file:
            groups[f.split('_')[0] if f[0].isalpha() else '自爬'].append(f)

    random.seed(0)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except Exception:
        font = ImageFont.load_default()

    max_edge = 1600
    for s in sorted(groups):
        f = random.choice(groups[s])
        img = Image.open(root / f).convert('RGB')
        W, H = img.size
        draw = ImageDraw.Draw(img)
        lw = max(2, min(W, H) // 500)
        for x, y, w, h, sc in boxes_by_file[f]:
            draw.rectangle([x, y, x + w, y + h], outline='red', width=lw)
            draw.text((x + lw, y + lw), f'{sc:.2f}', fill='yellow', font=font)
        if max(W, H) > max_edge:
            r = max_edge / max(W, H)
            img = img.resize((round(W * r), round(H * r)))
        out = REPO / f'{s}_sample.jpg'
        img.save(out, quality=88)
        print(f'{s:16s} {f} {W}x{H} {len(boxes_by_file[f])} 框 -> {out}')


def main():
    cod = json.load(open(COD_JSON))
    root = Path(cod['root'])
    images = cod['images']
    out_path = Path(OUT_JSONL)

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)['file'])
                except (json.JSONDecodeError, KeyError):
                    continue  # 崩溃残留的半行，忽略
    todo = [f for i, f in enumerate(images) if i % N_SHARDS == SHARD and f not in done]
    print(f'共 {len(images)} 图，本片 {len(todo)}（shard {SHARD}/{N_SHARDS}），已完 {len(done)}，输出 {out_path}')

    model = load_model(CKPT)

    n_boxes = 0
    with open(out_path, 'a', buffering=1) as fout, tqdm(todo, desc='Uni 召回', unit='图') as bar:
        for rel in todo:
            try:
                img = Image.open(root / rel).convert('RGB')
            except Exception as e:
                fout.write(json.dumps({'file': rel, 'missing': str(e)}, ensure_ascii=False) + '\n')
                bar.update(1)
                continue
            W, H, boxes = infer_image(model, img)
            fout.write(json.dumps({'file': rel, 'width': W, 'height': H, 'boxes': boxes},
                                  ensure_ascii=False) + '\n')
            n_boxes += len(boxes)
            bar.set_postfix(boxes=n_boxes)
            bar.update(1)

    print(f'完成：{out_path}，共 {n_boxes} 框（score>{SCORE_THRE}，跨 tile NMS 0.5）')


if __name__ == '__main__':
    viz(sys.argv[2] if len(sys.argv) > 2 else None) if len(sys.argv) > 1 and sys.argv[1] == 'viz' else main()
