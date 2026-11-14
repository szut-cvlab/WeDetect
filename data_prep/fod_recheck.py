#!/usr/bin/env python3
"""FOD 微调前后 Uni 召回对比：同一批 SkyFind 图，base / fod 两份 ckpt 各跑一遍。

输出  SkyFind/recheck_base.jsonl、SkyFind/recheck_fod.jsonl（逐图框，同 cod_uni_boxes.jsonl 格式）
      stdout 汇总对比（框量、>512px 大框占比、分数分布）
运行  python3 data_prep/fod_recheck.py            # 约 10 分钟，单卡
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cod_uni_recall import load_model, infer_image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
COD_JSON = Path('/mnt/sda1/datasets/drone/SkyFind/cod.json')
OUT_DIR = Path('/mnt/sda1/datasets/drone/SkyFind')
N_IMG = 50
CKPTS = [('fod', 'checkpoints/WeDetect/wedetect_large_uni_fod.pth'),
         ('fodcoco', 'checkpoints/WeDetect/wedetect_large_uni_fodcoco.pth')]
COMPARE = ['base', 'fod', 'fodcoco']  # base 是 NMS 0.7 旧结果，仅参考
BIG = 512  # 大框阈值（长边），与微调前统计口径一致


def main():
    import random
    cod = json.load(open(COD_JSON))
    root = Path(cod['root'])
    imgs = random.Random(0).sample(sorted(cod['images']), min(N_IMG, len(cod['images'])))
    print(f'抽样 {len(imgs)} 张（seed 0）')

    for tag, rel_ckpt in CKPTS:
        ckpt = REPO / rel_ckpt
        assert ckpt.exists(), f'缺少 ckpt: {ckpt}'
        out = OUT_DIR / f'recheck_{tag}.jsonl'
        model = load_model(ckpt)
        with open(out, 'w') as f:
            for rel in imgs:
                img = Image.open(root / rel).convert('RGB')
                W, H, boxes = infer_image(model, img)
                f.write(json.dumps({'file': rel, 'width': W, 'height': H, 'boxes': boxes},
                                   ensure_ascii=False) + '\n')
        del model
        torch.cuda.empty_cache()
        print(f'{tag}: {out}')

    print(f'\n{"":8}{"框/图":>10}{"大框占比":>12}{"分数中位":>12}{"大框分数中位":>14}')
    for tag in COMPARE:
        n_img = n_box = n_big = 0
        scores, big_scores = [], []
        for line in open(OUT_DIR / f'recheck_{tag}.jsonl'):
            r = json.loads(line)
            n_img += 1
            for x, y, w, h, s in r['boxes']:
                n_box += 1
                scores.append(s)
                if max(w, h) > BIG:
                    n_big += 1
                    big_scores.append(s)
        med = sorted(scores)[len(scores) // 2] if scores else 0
        big_med = sorted(big_scores)[len(big_scores) // 2] if big_scores else 0
        print(f'{tag:8}{n_box / n_img:>10.1f}{n_big / n_box:>12.1%}'
              f'{med:>12.4f}{big_med:>14.4f}')


if __name__ == '__main__':
    main()
