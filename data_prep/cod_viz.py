#!/usr/bin/env python3
"""每个数据源抽一张图，画上 Uni 召回的框，存到当前目录（<源>_sample.jpg）。"""
import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

COD_JSON = '/mnt/sda1/datasets/drone/SkyFind/cod.json'
BOXES_JSONL = '/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes.jsonl'
OUT_DIR = '/home/liuwei/projects/research/WeDetect'  # 样例图输出到仓库根，vscode 直接看
MAX_EDGE = 1600  # 输出长边压到 1600 方便看


def src(f):
    return f.split('_')[0] if f[0].isalpha() else '自爬'


def main():
    cod = json.load(open(COD_JSON))
    root = cod['root']
    boxes_by_file = {}
    for line in open(BOXES_JSONL):
        r = json.loads(line)
        if 'boxes' in r:
            boxes_by_file[r['file']] = r['boxes']

    groups = defaultdict(list)
    for f in cod['images']:
        if f in boxes_by_file:
            groups[src(f)].append(f)

    random.seed(0)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    except Exception:
        font = ImageFont.load_default()

    for s in sorted(groups):
        f = random.choice(groups[s])
        img = Image.open(os.path.join(root, f)).convert('RGB')
        W, H = img.size
        draw = ImageDraw.Draw(img)
        lw = max(2, min(W, H) // 500)
        for x, y, w, h, sc in boxes_by_file[f]:
            draw.rectangle([x, y, x + w, y + h], outline='red', width=lw)
            draw.text((x + lw, y + lw), f'{sc:.2f}', fill='yellow', font=font)
        if max(W, H) > MAX_EDGE:
            r = MAX_EDGE / max(W, H)
            img = img.resize((round(W * r), round(H * r)))
        out = os.path.join(OUT_DIR, f'{s}_sample.jpg')
        img.save(out, quality=88)
        print(f'{s:16s} {f} {W}x{H} {len(boxes_by_file[f])} 框 -> {out}')


if __name__ == '__main__':
    main()
