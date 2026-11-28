"""Tiny 域适配模型评测 MOBDrone test 考卷（1280 推理，5 类统一词表写法）。

mmdet test 输出 pkl 后用 eval/eval_ovd.py 离线算指标。
"""
_base_ = ["wedetect_tiny_ovd_finetune.py"]

import json

ANN = "/mnt/sda1/datasets/drone/coco/mobdrone_test.json"
_cats = json.load(open(ANN))["categories"]
CLASSES = [c["name"] for c in _cats]

model = dict(num_test_classes=len(CLASSES))

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=(1280, 1280)),
    dict(
        type="WeDetectLetterResize",
        scale=(1280, 1280),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(type="LoadText"),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "pad_param", "texts")),
]

test_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        _delete_=True,
        type="WeCocoDataset",
        data_root="/mnt/sda1/datasets/drone",
        ann_file="coco/mobdrone_test.json",
        data_prefix=dict(img=""),
        test_mode=True,
        batch_shapes_cfg=None,
        metainfo=dict(classes=CLASSES)),
    class_text_path="data/texts/mobdrone_ovd_class_texts.json",
    pipeline=test_pipeline)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=test_dataset)

test_evaluator = dict(type="CocoMetric", ann_file=ANN, metric="bbox")
