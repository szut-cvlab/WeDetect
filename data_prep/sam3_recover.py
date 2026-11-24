#!/usr/bin/env python3
"""SAM3 概念补漏：每图喂该图已有框出现过的 fine 概念词（通常几个到十几个），text-prompt SAM3 出框。
超过 SAM3 原生分辨率（1008）才切 tile；候选框先跨 tile/跨概念 NMS 0.5 得整图结果，
再与图中已有框 IoU 0.5 去重，留下的作为新增框。概念词本身已是归一化标签，新增框无需再清洗。
（只补图中已有类型的漏检实例；图中完全没出现过的类型不在此法覆盖范围）
输出  named_norm_recover.jsonl 只写 SAM3 新增行，追加写：
      {file, i, box:[x,y,w,h], score, label}；label 是单一 fine 词。
      i 从该图已有框 max(i)+1 续编。最终框全集 = JSONL_IN + 本文件，转 COCO 时拼接
断点  输出里有行的 file 跳过，中断重跑自动续
运行  一条命令自动按 GPUS 列表多卡并行（主进程分片起子进程，分片互斥；中断重跑同一命令自动续）：
      .venv-sam3/bin/python data_prep/sam3_recover.py
调试  test_one('000811.jpg') 单图试跑，只打印不落盘，存绿框可视化图
"""
import os
import json
import math
import sys
import subprocess
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / 'checkpoints/sam3'
ROOT = Path('/mnt/sda1/datasets/drone/SkyFind/train')
JSONL_IN = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3_named_norm.jsonl')   # 框/概念来源（也是全量 todo 来源）
JSONL_OUT = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3_named_norm_recover.jsonl')
SCORE_THRE = 0.5     # 概念实例分数阈值
NMS_THRE = 0.5       # 跨 tile/概念候选 NMS
DEDUP_THRE = 0.3     # 与已有框去重阈值（收紧到 0.3：轻微错位的重复检出也剔掉）
TILE = 1008          # SAM3 原生分辨率（processor target_size），超过才切 tile
BATCH = 16            # 每次 forward 的 (概念×tile) 对数，15.5G 卡 32 会 OOM
GPUS = [0, 1, 2, 3]         # 用哪几张卡：主进程给列表里每张卡起一个子进程，空闲就改这里


def tile_starts(edge):
    if edge <= TILE:
        return [0]
    n = math.ceil(edge / TILE)
    stride = (edge - TILE) / (n - 1)
    starts = sorted({round(i * stride) for i in range(n)})
    starts[-1] = edge - TILE
    return starts


def make_tiles(img):
    """→ [(x0, y0, tile_img)]"""
    W, H = img.size
    if W <= TILE and H <= TILE:
        return [(0, 0, img)]
    return [(x0, y0, img.crop((x0, y0, min(x0 + TILE, W), min(y0 + TILE, H))))
            for y0 in tile_starts(H) for x0 in tile_starts(W)]


def detect_chunk(proc, model, pairs):
    """一批 (concept, x0, y0, tile_img) 前向 → [(box_xyxy_图内, score, concept, x0, y0, tw, th)]；
    不出 mask（补框只要框）"""
    with torch.no_grad():
        inputs = proc(images=[p[3] for p in pairs], text=[p[0] for p in pairs],
                      return_tensors='pt').to('cuda')
        out = model(**inputs)
    scores = out.pred_logits.cpu().sigmoid()
    if out.presence_logits is not None:
        scores = scores * out.presence_logits.cpu().sigmoid()
    boxes = out.pred_boxes.cpu()   # (B, N, 4) xyxy 归一化
    out_pairs = []
    for bi, (concept, x0, y0, tile) in enumerate(pairs):
        s = scores[bi]
        for j in (s > SCORE_THRE).nonzero(as_tuple=True)[0].tolist():
            bw, bh = tile.size
            x1, y1, x2, y2 = (boxes[bi, j] * torch.tensor([bw, bh, bw, bh])).tolist()
            out_pairs.append(([x1 + x0, y1 + y0, x2 + x0, y2 + y0], float(s[j]),
                              concept, x0, y0, bw, bh))
    return out_pairs


def recover_image(proc, model, entry, img):
    """单图补漏：entry = {'xyxy': 已有框, 'labels': 本地概念} → [(box_xywh, score, label)] 按分排序"""
    W, H = img.size
    concepts = sorted(entry['labels'])
    tiles = make_tiles(img)
    pairs = [(c, t[0], t[1], t[2]) for c in concepts for t in tiles]
    cand_raw = []
    for k in range(0, len(pairs), BATCH):
        cand_raw += detect_chunk(proc, model, pairs[k:k + BATCH])
    # 裁回图内；贴 tile 切线的截断框删掉（重叠区会在相邻 tile 完整重检，贴图边缘的保留）
    EPS = 2
    cand = []
    for (x1, y1, x2, y2), s, c, x0, y0, tw, th in cand_raw:
        x1, y1, x2, y2 = min(max(x1, 0), W), min(max(y1, 0), H), min(x2, W), min(y2, H)
        if x2 <= x1 or y2 <= y1:
            continue
        if (x0 > 0 and x1 <= x0 + EPS) or (x0 + tw < W and x2 >= x0 + tw - EPS) or \
           (y0 > 0 and y1 <= y0 + EPS) or (y0 + th < H and y2 >= y0 + th - EPS):
            continue
        cand.append(([x1, y1, x2, y2], s, c))
    new_boxes = []
    if cand:
        keep = torchvision.ops.nms(torch.tensor([c[0] for c in cand]),
                                   torch.tensor([c[1] for c in cand]), NMS_THRE).tolist()
        e_xyxy = torch.tensor(entry['xyxy'])
        for k in keep:
            (x1, y1, x2, y2), s, c = cand[k]
            iou = torchvision.ops.box_iou(torch.tensor([[x1, y1, x2, y2]]), e_xyxy)[0]
            if float(iou.max()) <= DEDUP_THRE:   # 与已有框去重
                new_boxes.append(([round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
                                  round(s, 4), c))
    new_boxes.sort(key=lambda x: -x[1])
    return new_boxes


def vis(img, exist_xyxy, new_boxes, name):
    """已有框红色、SAM3 新增框绿色带标签，存工作区根目录"""
    draw = ImageDraw.Draw(img)
    lw = max(2, min(img.size) // 1000)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    except Exception:
        font = None
    for x1, y1, x2, y2 in exist_xyxy:
        draw.rectangle([x1, y1, x2, y2], outline='red', width=lw * 2)
    for b, s, label in new_boxes:
        x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=lw * 2)
        draw.text((x1 + lw, y1 + lw), f'{label} {s:.2f}', fill='lime', font=font)
    img.resize((1600, int(1600 * img.size[1] / img.size[0]))).save(
        REPO / f'sam3_recover_{Path(name).stem}.jpg', quality=85)


def test_one(file='000811.jpg', concepts=None):
    """单图调试：跑一张图的概念补漏，只打印不落盘，存可视化图（红=已有框，绿=SAM3 新增）。"""
    proc = Sam3Processor.from_pretrained(CKPT)
    model = Sam3Model.from_pretrained(CKPT).cuda().eval()
    entry = {'xyxy': [], 'labels': set()}
    for line in open(JSONL_IN):
        r = json.loads(line)
        if r['file'] == file:
            x1, y1, w, h = r['box'][:4]
            entry['xyxy'].append([x1, y1, x1 + w, y1 + h])
            entry['labels'].add(r['label'])
    if not entry['xyxy']:
        print(f'{file} 在 {JSONL_IN.name} 里没有框')
        return
    if concepts:
        entry['labels'] &= {c.strip().lower() for c in concepts}
    img = Image.open(ROOT / file).convert('RGB')
    new_boxes = recover_image(proc, model, entry, img)
    for b, s, c in new_boxes:
        print(f'{c} {s:.2f} box={b}')
    print(f'共 {len(new_boxes)} 个新增框（阈值 {SCORE_THRE}，NMS {NMS_THRE}）')
    vis(img, entry['xyxy'], new_boxes, file)


def main(files, gpu):
    """子进程：跑分片给定的图列表（主进程已按 done 过滤，分片互斥）"""
    # 每图概念（该图已有框的 label 集合）+ 已有框，一次扫描
    exist = {}
    for line in open(JSONL_IN):
        r = json.loads(line)
        x1, y1, w, h = r['box'][:4]
        e = exist.setdefault(r['file'], {'xyxy': [], 'max_i': -1, 'labels': set()})
        e['xyxy'].append([x1, y1, x1 + w, y1 + h])
        e['max_i'] = max(e['max_i'], r['i'])
        e['labels'].add(r['label'])

    print(f'GPU {gpu}: 本次 {len(files)} 张，输出 {JSONL_OUT}', flush=True)
    proc = Sam3Processor.from_pretrained(CKPT)
    model = Sam3Model.from_pretrained(CKPT).cuda().eval()
    with open(JSONL_OUT, 'a') as fout, tqdm(files, desc=f'gpu{gpu}', unit='img') as pbar:
        for name in pbar:
            img = Image.open(ROOT / name).convert('RGB')
            new_boxes = recover_image(proc, model, exist[name], img)
            max_i = exist[name]['max_i']
            for k, (box, s, c) in enumerate(new_boxes):
                fout.write(json.dumps({'file': name, 'i': max_i + 1 + k, 'box': box,
                                       'score': s, 'label': c}, ensure_ascii=False) + '\n')
            fout.flush()


if __name__ == '__main__':
    # test_one('000811.jpg')
    if len(sys.argv) == 3:      # 子进程: sam3_recover.py <gpu> <分片列表文件>
        main(open(sys.argv[2]).read().split(), int(sys.argv[1]))
    else:                       # 主进程: 扫描全量图 → 滤已跑 → 分片 → 每卡起一个子进程
        done = set()
        if JSONL_OUT.exists():
            for line in open(JSONL_OUT):
                done.add(json.loads(line)['file'])
        files = set()
        for line in open(JSONL_IN):
            files.add(json.loads(line)['file'])
        todo = sorted(files - done)
        print(f'图 {len(files)} 张（输出已有 {len(done)}），本次 {len(todo)} 张，'
              f'{len(GPUS)} 卡并行: {GPUS}', flush=True)
        procs = []
        for k, g in enumerate(GPUS):
            lst = f'/tmp/sam3_recover_gpu{g}.txt'
            open(lst, 'w').write('\n'.join(todo[k::len(GPUS)]))
            procs.append(subprocess.Popen([sys.executable, __file__, str(g), lst],
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))))
        for p in procs:
            p.wait()
        bad = [(p.args[2], p.returncode) for p in procs if p.returncode]
        print(f'子进程退出码: {[p.returncode for p in procs]}'
              + (f'，异常: {bad}' if bad else '，全部正常'), flush=True)
