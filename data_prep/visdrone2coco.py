#!/usr/bin/env python
"""VisDrone DET train+val 原始标注 -> COCO JSON。

数据位置
    图像  /mnt/sda1/datasets/drone/VisDrone/images/{train,val}
          目录里还有同名 .npy（早前某条管线留下），只取 .jpg
    标注  VisDrone2019-DET-{train,val}/annotations/*.txt
          每行 8 字段逗号分隔：left,top,width,height,score,category,truncate,occlusion
          类别编号 0 到 11：0 ignored regions、1 到 10 检测类、11 others
    产物  /mnt/sda1/datasets/drone/coco/visdrone_trainval.json 与 visdrone_class_texts.json
          类名表格式同仓库 data/texts/visdrone_en_class_texts.json，即 [["pedestrian"], ...]
          file_name 一律相对 /mnt/sda1/datasets/drone，形如 VisDrone/images/train/xxx.jpg，
          这样所有集的 json 共用一个 data_root（池内约定，见 notes/2026-11-07 §2.1）

为什么不用 labels/ 那套 YOLO txt
    那是早前从官方标注转出来的，转的时候把 category=0、category=11 和所有 score=0 的行
    删掉了，截断与遮挡标记也丢了。现在一律以 annotations/ 原件为准。

三类行的处理
    1 到 10  写进 annotations，bbox 转像素 [x,y,w,h] 并夹到图内，保留 score/truncate/occlusion
    0        进 images[].ignore_regions
    11       进 images[].others_regions
    score=0 只出现在 0 与 11 两类，所以 1 到 10 的框全部参与评测，没有「被排除的难点目标」。

关于 ignore：训练代码不处理，这两个字段只是留档
    ignore_regions 与 others_regions 挂在 image 条目上，mmdet 只读 images 的
    id/file_name/width/height，自定义字段直接忽略，所以这 11,753 个区域训练时当背景。
    链路上另两头也是断的：wedetect/datasets/wdscoco.py:152 把 ignore_flag 写死 0；
    YOLOWorldHead.loss_by_feat 收了 batch_gt_instances_ignore 参数但函数体没用
    （仓库里只有 yolov5_head.py 的 _loss_by_feat_with_ignore 真做屏蔽，默认 ignore_iof_thr=-1 关着）。
    决定：暂不屏蔽，但代价要记清（实测）：
      ignored 10,189 个，中位边长 24px，36% 大于 32px，总面积相当于有效类标注面积的 4.4%
      ——这些是成片的密集人群与遮挡堆，被当背景是实打实的假负，不是可以忽略的碎片
      others 1,564 个，中位边长 32px，面积占 0.7%，FOD 补标第一轮会被 VLM 命名成真类
    等补标轮做完，若 AP 明显受压制，再回来改 loss 侧的屏蔽。

另记：VisDrone 原生标注不穷尽，小目标漏标多，入池前需补标审计，否则漏标区域成假负样本。
"""
import json, os
from collections import Counter
from PIL import Image

ROOT = "/mnt/sda1/datasets/drone/VisDrone"
DRONE_ROOT = "/mnt/sda1/datasets/drone"
OUT_DIR = f"{DRONE_ROOT}/coco"
SPLITS = ["train", "val"]
# 类名规则：只用空格分隔，连字符与下划线一律换掉（不改写词义）。
# 官方 awning-tricycle 写成 awning tricycle；两者文本 embedding 余弦 0.980，
# 精度无差，归一只是为了合并词表时同一概念不再裂成两个类（check_class_tokenizer.py）。
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning tricycle", "bus", "motor"]
images, anns = [], []
cats = [{"id": i, "name": n, "supercategory": ""} for i, n in enumerate(CLASSES)]
aid = 0
stat = Counter()
for sp in SPLITS:
    img_dir = f"{ROOT}/images/{sp}"
    ann_dir = f"{ROOT}/VisDrone2019-DET-{sp}/annotations"
    for fn in sorted(os.listdir(img_dir)):
        if not fn.endswith(".jpg"):
            continue
        with Image.open(os.path.join(img_dir, fn)) as im:
            w, h = im.size
        iid = len(images)
        images.append({"id": iid, "file_name": f"VisDrone/images/{sp}/{fn}", "width": w, "height": h,
                       "ignore_regions": [], "others_regions": []})
        ap = os.path.join(ann_dir, fn[:-4] + ".txt")
        if not os.path.exists(ap):
            stat["no_anno_file"] += 1
            continue
        for line in open(ap):
            p = [v for v in line.strip().split(",") if v != ""]
            if len(p) != 8:
                stat["bad_line"] += 1
                continue
            left, top, ww, hh, score, c, trunc, occ = map(int, p)
            stat["lines"] += 1
            if ww <= 0 or hh <= 0:
                stat["empty_box"] += 1
                continue
            if c == 0:                      # ignored regions，官方 0 类
                stat["ignore"] += 1
                images[-1]["ignore_regions"].append([left, top, ww, hh])
                continue
            if c == 11:                     # others，官方 11 类，10 类检测任务不算它
                stat["others"] += 1
                images[-1]["others_regions"].append([left, top, ww, hh])
                continue
            if not 1 <= c <= len(CLASSES):
                stat["unknown_cat"] += 1
                continue
            x, y = max(0, left), max(0, top)
            ww2, hh2 = min(ww, w - x), min(hh, h - y)
            if ww2 <= 0 or hh2 <= 0:
                stat["clipped_out"] += 1
                continue
            if score == 0:
                stat["score0"] += 1
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": c - 1,
                         "bbox": [x, y, ww2, hh2], "area": ww2 * hh2,
                         "iscrowd": 0, "score": score,
                         "truncate": trunc, "occlusion": occ})

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/visdrone_trainval.json", "w") as f:
    json.dump({"images": images, "annotations": anns, "categories": cats}, f)
with open(f"{OUT_DIR}/visdrone_class_texts.json", "w") as f:
    json.dump([[c["name"]] for c in cats], f, ensure_ascii=False)

print(f"images={len(images)} boxes={len(anns)}")
for k, v in sorted(stat.items()):
    print(f"  {k:12s} {v}")
cnt = Counter(a["category_id"] for a in anns)
for i, n in enumerate(CLASSES):
    print(f"  {n:16s} {cnt[i]}")
print("  truncate", Counter(a["truncate"] for a in anns))
print("  occlusion", Counter(a["occlusion"] for a in anns))
side = [min(a["bbox"][2], a["bbox"][3]) for a in anns]
for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]:
    print(f"  edge [{lo},{hi}) = {sum(1 for s in side if lo <= s < hi)}")
# 路径校验：file_name 一律相对 /mnt/sda1/datasets/drone，逐条确认可达
missing = [im["file_name"] for im in images
           if not os.path.exists(os.path.join(DRONE_ROOT, im["file_name"]))]
print(f"  file_name 校验 missing={len(missing)}", missing[:3])
