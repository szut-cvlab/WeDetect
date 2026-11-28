"""Tiny 检测底座域适配：ovd_train 全参微调，文本塔 lr_mult=0.01 跟训，1280 切片输入。

配方沿 config/wedetect_base_coco_full_tuning_8xbs4_2e-5.py（COCO 全参微调），
数据换 ovd_train.json（FOD 七集 + COD，98,198 图 / 2,440,066 框），val 用 ovd_val.json
（3,030 图，考卷不参与调参）。数据口径见 notes/2026-11-07 §2.4。
"""
_base_ = ["wedetect_tiny.py"]

import json

VOCAB = json.load(open("/mnt/sda1/datasets/drone/coco/vocab_merged.json"))
num_classes = len(VOCAB)  # 178，assigner 与负类池；metainfo 顺序 = 词表 id 顺序
num_training_classes = 80  # 每样本 texts 上限（正类+负采样），YOLO-World 同款

load_from = "checkpoints/WeDetect/wedetect_tiny.pth"

base_lr = 2e-5
weight_decay = 0.05
train_batch_size_per_gpu = 2  # bs4 实测 OOM（无/有 AMP 都爆 16GB），只能 bs2
accumulative_counts = 8       # 2卡×bs2×8 = 有效 batch 32 = 官方 8xbs4
sync_bn = "torch"             # 63 个 BatchNorm2d：2卡×bs2 每卡统计只有 2 图，SyncBN 聚合后 4 = 官方单卡 4（mmengine 0.10 要字符串，不能 True）
max_epochs = 12
close_mosaic_epochs = 4
save_epoch_intervals = 1
persistent_workers = True

img_scale = (1280, 1280)
affine_scale = 0.5
mixup_prob = 0.15

model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    bbox_head=dict(head_module=dict(num_classes=num_training_classes)),
    # assigner num_classes = 每样本 texts 数（head 输出维数），非词表大小
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    # 密集场景 NMS top-300 打满（VisDrone 84% 图），统一 500 口径与 eval_ovd 对齐
    test_cfg=dict(max_per_img=500),
)

# ---- train pipeline（1280 版，含 RandomLoadText 在线词表） ----
pre_transform = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
]

text_transform = [
    dict(
        type="RandomLoadText",
        num_neg_samples=(num_classes, num_classes),
        max_num_samples=num_training_classes,
        padding_to_max=True,
        padding_value=""),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "flip",
                   "flip_direction", "texts")),
]

mosaic_affine_transform = [
    dict(
        type="MultiModalMosaic",
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type="WeDetectRandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

albu_train_transforms = [
    dict(type="Blur", p=0.01),
    dict(type="MedianBlur", p=0.01),
    dict(type="ToGray", p=0.01),
    dict(type="CLAHE", p=0.01),
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type="YOLOv5MultiModalMixUp",
        prob=mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    dict(
        type="mmdet.Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"]),
        keymap={"img": "image", "gt_bboxes": "bboxes"}),
    dict(type="WeDetectHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    *text_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type="WeDetectRandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(
        type="mmdet.Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"]),
        keymap={"img": "image", "gt_bboxes": "bboxes"}),
    dict(type="WeDetectHSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    *text_transform,
]

default_hooks = dict(logger=dict(type="LoggerHook", interval=200))
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=200)

custom_hooks = [
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2),
]

# ---- 数据 ----
train_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root="/mnt/sda1/datasets/drone",
        ann_file="coco/ovd_train.json",
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        # 必须显式传：父类默认 metainfo 是 COCO 80 类，ann categories 顺序 = 词表顺序
        metainfo=dict(classes=VOCAB)),
    class_text_path="/mnt/sda1/datasets/drone/coco/ovd_train_class_texts.json",
    pipeline=train_pipeline)

train_dataloader = dict(
    num_workers=8,
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type="yolow_collate"),
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset)

# ---- val/test：ovd_val.json，1280 推理 ----
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="WeDetectKeepRatioResize", scale=img_scale),
    dict(
        type="WeDetectLetterResize",
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type="LoadAnnotations", with_bbox=True, _scope_='mmdet'),
    dict(type="LoadText"),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                   "scale_factor", "pad_param", "texts")),
]

val_dataset = dict(
    type="MultiModalDataset",
    dataset=dict(
        type="WeCocoDataset",
        data_root="/mnt/sda1/datasets/drone",
        ann_file="coco/ovd_val.json",
        data_prefix=dict(img=""),
        test_mode=True,
        batch_shapes_cfg=None,
        metainfo=dict(classes=VOCAB)),
    class_text_path="/mnt/sda1/datasets/drone/coco/ovd_val_class_texts.json",
    pipeline=test_pipeline)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file="/mnt/sda1/datasets/drone/coco/ovd_val.json",
    metric="bbox")
test_evaluator = val_evaluator

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# ---- 训练设置（照 COCO 全参配方） ----
param_scheduler = [
    # warmup 对齐官方占比（1000/3697≈27% 首 epoch）：本配方首 epoch 24550 microstep×27%≈6600
    dict(type="LinearLR", start_factor=0.001, end_factor=1.0,
         begin=0, end=6600, by_epoch=False),
    dict(type="LinearLR", start_factor=1.0, end_factor=0.001,
         begin=0, end=max_epochs, by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])

optim_wrapper = dict(
    type="OptimWrapper",
    clip_grad=dict(max_norm=10.0),
    accumulative_counts=accumulative_counts,
    optimizer=dict(
        type="AdamW",
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={"backbone.text_model": dict(lr_mult=0.01),
                     "logit_scale": dict(weight_decay=0.0)}),
    constructor="YOLOWv5OptimizerConstructor")
