#!/usr/bin/env python3
"""框全集合并：named_norm.jsonl（VLM 命名+清洗）+ recover.jsonl（SAM3 补漏）→ 全量框文件。
统一行格式 {file, i, box, score, label}，label = 训练类名（fine，已含回退 mid 与词表清洗）。
<4px 框救援：named_norm 的 <4px 行回溯 Uni 原框（named box == sam3.jsonl box 精确匹配，
score 从 Uni 沿用可配对），Uni 原框 ≥4px 的用原框几何替换（recheck 紧框收缩误伤的救回）；
Uni 原框也 <4px 的与 recover 的 <4px 框（SAM3 自检，从未收缩）剔除。
类别表按类名排序固定 id，落盘 json，训练/评测必须共用同一张。
运行  .venv/bin/python data_prep/merge_boxes.py
"""
import json
from collections import Counter
from pathlib import Path

DIR = Path('/mnt/sda1/datasets/drone/SkyFind')
SRC_A = DIR / 'cod_uni_boxes_fodcoco_sam3_named_norm.jsonl'
SRC_B = DIR / 'cod_uni_boxes_fodcoco_sam3_named_norm_recover.jsonl'
SRC_S3 = DIR / 'cod_uni_boxes_fodcoco_sam3.jsonl'
SRC_UNI = DIR / 'cod_uni_boxes_fodcoco.jsonl'
OUT = DIR / 'cod_uni_boxes_fodcoco_sam3_named_norm_all.jsonl'
CLASSES = DIR / 'cod_uni_boxes_fodcoco_sam3_named_norm_all_classes.jsonl'
N_A, N_B = 499506, 131323   # 预期行数，不符即中止
N_RESCUE = 537              # 预期救援数（Uni 原框 ≥4px 的 <4px 行）


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        assert set(r) >= {'file', 'i', 'box', 'label'} and len(r['box']) >= 4, path
        rows.append({'file': r['file'], 'i': r['i'], 'box': r['box'][:4],
                     'score': r['score'], 'label': r['label']})
    return rows


def uni_original(r, s3, uni):
    """named_norm 的框 → Uni 原框。named box 与 sam3.jsonl box 逐值相同（紧框替换），
    sam3 的 score 沿用 Uni，按分数配对、同分取位置最近。找不到返回 None。"""
    f, b, sc = r['file'], r['box'], r['score']
    hit = [u for u in s3.get(f, []) if u[:4] == b and abs(u[4] - sc) < 1e-6]
    if not hit:
        return None
    cands = [u for u in uni.get(f, []) if abs(u[4] - hit[0][4]) < 1e-6]
    if not cands:
        return None
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return min(cands, key=lambda v: (v[0] + v[2] / 2 - cx) ** 2 + (v[1] + v[3] / 2 - cy) ** 2)


def main():
    a, b = load(SRC_A), load(SRC_B)
    assert len(a) == N_A and len(b) == N_B, (len(a), len(b))
    # <4px 框：能回溯到 ≥4px Uni 原框的救援，其余剔除
    files = {r['file'] for r in a if r['box'][2] < 4 or r['box'][3] < 4}
    s3, uni = {}, {}
    for path, store in ((SRC_S3, s3), (SRC_UNI, uni)):
        for line in open(path):
            r = json.loads(line)
            if r['file'] in files:
                store[r['file']] = r.get('boxes', [])
    rescued = 0
    rows = []
    for r in a:
        if r['box'][2] >= 4 and r['box'][3] >= 4:
            rows.append(r)
            continue
        u = uni_original(r, s3, uni)
        if u and min(u[2], u[3]) >= 4:
            rows.append(dict(r, box=[round(u[0]), round(u[1]), round(u[2]), round(u[3])]))
            rescued += 1
    for r in b:
        if r['box'][2] >= 4 and r['box'][3] >= 4:
            rows.append(r)
    print(f'<4px 剔除 {N_A + N_B - len(rows)}，救援 {rescued}')
    assert rescued == N_RESCUE, rescued
    seen = set()
    for r in rows:
        assert r['box'][2] >= 4 and r['box'][3] >= 4, r
        k = (r['file'], r['i'])
        assert k not in seen, k
        seen.add(k)
    # 类别表：按类名排序固定 id
    vocab = sorted({r['label'] for r in rows})
    assert len(vocab) == 288, len(vocab)
    with open(OUT, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(CLASSES, 'w') as f:
        json.dump({w: i for i, w in enumerate(vocab)}, f, ensure_ascii=False, indent=1)
    top = Counter(r['label'] for r in rows).most_common(6)
    print(f'合并 {len(a)} + {len(b)} = {len(rows)} 行，{len(vocab)} 类，已写 {OUT}')
    print(f'类别表已写 {CLASSES}')
    print('top6:', ' / '.join(f'{w} {n}' for w, n in top))


if __name__ == '__main__':
    main()
