#!/usr/bin/env python3
"""从 COCO train2017 取"全部标注框都是白名单类别"的图做 FOD objectness 微调 replay：类别并 0。

严格口径：图内只要有一个非白名单框（室内类等）就整张丢弃——零矛盾标注，且这些图
天然是纯室外场景，域匹配最好。

输出  /mnt/sda1/datasets/coco/annotations/coco_replay_objectness.json
      file_name 形如 train2017/xxx.jpg（相对 /mnt/sda1/datasets/coco）
运行  python3 data_prep/make_coco_replay.py
"""
import json
from collections import Counter

SRC = '/mnt/sda1/datasets/coco/annotations/instances_train2017.json'
OUT = '/mnt/sda1/datasets/coco/annotations/coco_replay_objectness.json'
# 只留无人机俯拍下会出现的类别：交通工具/交通设施/动物，与 FOD 词表重合
# （COCO 的室内 80 类——家具厨具电器日用品——场景污染航拍 objectness 先验）
ALLOWED = {'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
           'truck', 'boat', 'traffic light', 'stop sign', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}

d = json.load(open(SRC))

name_by_id = {c['id']: c['name'] for c in d['categories']}
ann_by_img = {}
for an in d['annotations']:
    ann_by_img.setdefault(an['image_id'], []).append(an)
# 严格口径：图内全部框都是白名单类才收
strict = {iid: lst for iid, lst in ann_by_img.items()
          if all(name_by_id[an['category_id']] in ALLOWED for an in lst)}
print(f'严格口径图: {len(strict)}/{len(ann_by_img)}（train2017 总图 {len(d["images"])}）')

images, anns = [], []
sizes = Counter()
cat_count = Counter()
img_by_id = {im['id']: im for im in d['images']}
for iid, ann_list in strict.items():
    im = dict(img_by_id[iid], file_name=f"train2017/{img_by_id[iid]['file_name']}")
    images.append(im)
    for an in ann_list:
        x, y, w, h = an['bbox']
        sizes['<=64' if max(w, h) <= 64 else '65-256' if max(w, h) <= 256 else '>256'] += 1
        cat_count[name_by_id[an['category_id']]] += 1
        anns.append({'id': len(anns) + 1, 'image_id': an['image_id'], 'category_id': 0,
                     'bbox': [x, y, w, h], 'area': an['area'], 'iscrowd': an['iscrowd']})

out = {'images': images, 'annotations': anns,
       'categories': [{'id': 0, 'name': 'object', 'supercategory': 'object'}]}
json.dump(out, open(OUT, 'w'))
print(f'replay: {len(images)} 图 {len(anns)} 框 -> {OUT}')
print('目标长边分布:', dict(sizes))
print('逐类框数:', dict(cat_count.most_common()))
