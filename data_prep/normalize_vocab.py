#!/usr/bin/env python3
"""训练标签清洗：训练类名 label = fine（部件/背景词或缺则回退 mid，mid 也脏则整框剔除）。
三件事：JUNK 白名单剔除（唯一删除来源）、同义词改名（mid/fine 各一张表，只列需要改的）、
motor 按 VLM 原 coarse 分流（vehicle→motorcycle / boat→motorboat）。
输出 <原名>_norm.jsonl，只写有标签的行（拒识/JUNK 行不写，记录留在 named.jsonl 可追溯），
每行新增 label 字段 = 最终训练类名，coarse/mid/fine 原样保留。
运行  .venv/bin/python data_prep/normalize_vocab.py
"""
import json
from collections import Counter
from pathlib import Path

JSONL_NAMED = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3_named.jsonl')
JSONL_OUT = JSONL_NAMED.with_name(JSONL_NAMED.stem + '_norm.jsonl')

# 剔除白名单（唯一删除来源），fine/mid 两层都查：fine 命中则回退 mid，mid 也命中才剔除
# 词表来自人工分桶：物体部件（arm/leg/window…）、背景 stuff（tree/dock/bridge…）、泛词乱码（quadruped/motorc…）
JUNK = {'arm', 'leg', 'hand', 'head', 'window', 'wall', 'door', 'wood', 'log', 'bush',
        'tree', 'plant', 'flower', 'dock', 'pool', 'bridge', 'deck', 'water tank', 'tank',
        'container', 'solar panel', 'satellite dish', 'bamboo', 'truss', 'pole', 'railing',
        'pipe', 'chimney', 'vent', 'silo', 'fly', 'quadruped', 'other vehicle', 'machine',
        'inflatable', 'ac_unit', 'light',
        'foot', 'propeller', 'footprint', 'shadow', 'debris', 'island', 'cliff', 'balcony',
        'roof', 'brick', 'stack', 'manhole', 'manhole cover', 'antenna', 'flagpole',
        'canopy', 'awning', 'palm', 'shrub', 'structure', 'other', 'unicorn', 'galaxy',
        'bicycle seat', 'equipment box', 'cargo container', 'ac unit', 'float',
        'fence', 'barrier', 'light pole',
        # 建筑家族整体剔除（2026-11-23）：俯拍下 house/building 等类间边界无判据、标注覆盖任意、
        # SAM3 recover 对 house 粒度漂移（单栋/整排混框），作为背景 stuff 处理
        'house', 'building', 'apartment', 'shed', 'warehouse',
        'cabin', 'hut', 'church', 'tower', 'factory'}

# mid 层同义词/拼写修复（只列需要改的）
RENAME_MID = {
    'motor': 'motorcycle', 'motor bike': 'motorcycle', 'motorc': 'motorcycle',
    'bike': 'bicycle', 'train car': 'train',
    'rv': 'van', 'caravan': 'van', 'van/mini-van': 'van',
    'suv': 'car', 'dump truck': 'truck',
    'bicyclist': 'cyclist', 'bicycle rider': 'cyclist',
    'cattle': 'cow', 'motor boat': 'motorboat', 'aircraft': 'airplane',
    'road_sign': 'road sign', 'streetlight': 'street light',
    'sports_equipment': 'sports equipment',
}

# fine 层同义词/拼写修复（只列需要改的）
RENAME_FINE = {
    'motor': 'motorcycle', 'motor boat': 'motorboat', 'motorc': 'motorcycle',
    'semi': 'semi-trailer truck', 'semi-truck': 'semi-trailer truck',
    'semi truck': 'semi-trailer truck', 'semi trailer': 'semi-trailer truck',
    'semi trailer truck': 'semi-trailer truck', 'semi-trailer': 'semi-trailer truck',
    'sports utility vehicle': 'suv', 'sports_equipment': 'sports equipment',
    'road_sign': 'road sign', 'cattle': 'cow', 'train car': 'train',
    'bicyclist': 'cyclist', 'aircraft': 'airplane', 'van/mini-van': 'van',
    'streetlight': 'street light',
    'cargo_ship': 'cargo ship', 'cargoship': 'cargo ship', 'fighter_jet': 'fighter jet',
    'passenger_ship': 'passenger ship', 'amusement_ride': 'amusement ride',
    'street_sign': 'street sign', 'speed_limit_sign': 'speed limit sign',
    'clothes_drying_rack': 'clothes drying rack',
    'speedboat': 'motorboat', 'motorcutter': 'motorboat', 'motorcruiser': 'motorboat',
    'sailing ship': 'sailboat',
    'minivan': 'van', 'camper van': 'van', 'motorhome': 'van', 'camper': 'van',
    'mobile home': 'van',
    'flatbed': 'flatbed truck', 'pickup truck': 'pickup', 'car carrier truck': 'car carrier',
    'container truck': 'semi-trailer truck', 'tanker': 'tanker truck',
    'freight car': 'train', 'container car': 'train', 'boxcar': 'train', 'tank car': 'train',
    'tanker car': 'train', 'railcar': 'train', 'container train car': 'train',
    'freight train': 'train', 'commuter train': 'train', 'container train': 'train',
    'city bus': 'bus', 'shuttle bus': 'bus', 'double bus': 'bus',
    'runner': 'person', 'sitter': 'person', 'sitting person': 'person',
    'dalmatian': 'dog', 'polar bear': 'bear',
    'combine harvester': 'harvester', 'forage harvester': 'harvester',
    'cement mixer': 'concrete mixer', 'mobile crane': 'crane', 'crane truck': 'crane',
    'front loader': 'loader', 'wheel loader': 'loader', 'skid steer loader': 'loader',
    'backhoe loader': 'loader',
    'travel trailer': 'trailer', 'flatbed trailer': 'trailer',
    'bicycle rider': 'cyclist', 'rv': 'van', 'caravan': 'van',
    'off road vehicle': 'off-road vehicle', 'forklift': 'forklift truck',
    'tank truck': 'tanker truck',
}

# 一词多义按 VLM 原 coarse 分流（vehicle 语境的 motor 走 RENAME_MID 默认 → motorcycle）
RENAME_BY_COARSE = {
    ('motor', 'boat'): 'motorboat',
}


def clean(w, table, coarse):
    w0 = str(w or '').strip().lower()
    w = table.get(w0, w0)
    return RENAME_BY_COARSE.get((w0, coarse), w)  # 一词多义按"原始词+coarse"分流，必须在改名后仍能回查原词


def label_of(r):
    """→ (训练类名 or None, 被剔的词)。fine 优先，脏/缺回退 mid，mid 也脏则剔除"""
    coarse = str(r.get('coarse') or '').strip().lower()
    fine = clean(r.get('fine'), RENAME_FINE, coarse)
    mid = clean(r.get('mid'), RENAME_MID, coarse)
    label = fine if fine and fine not in JUNK else mid
    if not label or label in JUNK:
        return None, (fine or mid) or '(空)'
    return label, None


def main():
    stats = Counter()
    dropped = Counter()
    labels = Counter()
    with open(JSONL_OUT, 'w') as fout:
        for line in open(JSONL_NAMED):
            r = json.loads(line)
            if r.get('reject'):
                stats['in_reject'] += 1
                continue  # 拒识行不进 _norm，留在 named.jsonl 可追溯
            stats['in_ok'] += 1
            label, bad = label_of(r)
            if label is None:
                stats['drop'] += 1
                dropped[bad] += 1
                continue  # JUNK 剔除行同样不进 _norm
            labels[label] += 1
            stats['out_ok'] += 1
            fout.write(json.dumps({'file': r['file'], 'i': r['i'], 'box': r['box'],
                                   'score': r['score'], 'coarse': r.get('coarse'),
                                   'mid': r.get('mid'), 'fine': r.get('fine'),
                                   'label': label}, ensure_ascii=False) + '\n')

    print(f"输入成功 {stats['in_ok']} / 拒识 {stats['in_reject']}；"
          f"有标签 {stats['out_ok']}，剔除 {stats['drop']}（不落盘）")
    print(f"\n== 训练类名（{len(labels)} 个） top40 ==")
    print('  ' + ', '.join(f'{w} {n}' for w, n in labels.most_common(40)))
    print(f"\n== 剔除 {len(dropped)} 词 ==")
    print('  ' + ', '.join(f'{w}({n})' for w, n in dropped.most_common()))
    print(f"\n已写 {JSONL_OUT}，共 {stats['out_ok']} 行")


if __name__ == '__main__':
    main()
