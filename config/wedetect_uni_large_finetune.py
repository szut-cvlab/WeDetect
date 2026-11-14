# Uni Large 域适配微调：底座冻结，训 neck/head/objectness prompt
# 数据：FOD 五集全图（类别并 0，objectness 格式），1280 输入
_base_ = ["wedetect_uni_base_finetune.py"]

load_from = 'checkpoints/WeDetect/wedetect_large_uni.pth'

# 模型尺寸 base→large：底座、neck 通道（scale 1.0→1.5）、head
model = dict(
    backbone=dict(image_model=dict(model_name='large')),
    neck=dict(scale_factor=1.5, model_size='large'),
    bbox_head=dict(head_module=dict(model_size='large')),
)

# 1280 输入（与 Large-Uni 推理一致），父配置的 pipeline 在解析时已按 640 固化，全部重建
img_scale = (1280, 1280)
train_batch_size_per_gpu = 2  # 1280 大图，实测 bs2 峰值约 12GB/16GB，bs4 反推约 20GB 会 OOM

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='WeDetectRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - 0.5, 1 + 0.5),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114))
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=0.15,
        pre_transform=[*pre_transform,
                       *mosaic_affine_transform]),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='WeDetectKeepRatioResize', scale=img_scale),
    dict(
        type='WeDetectLetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='WeDetectRandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='WeDetectHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction'))
]

# 覆盖 base 的 custom_hooks：其 switch_pipeline 在 base 解析时固化为 640，必须重挂 1280 版
custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=2,  # max_epochs(4) - close_mosaic_epochs(2)
        switch_pipeline=train_pipeline_stage2)
]

# FOD 五集 objectness（data_prep/fod_objectness.py 生成，category_id 全并 0）
fod_dataset = dict(
    type='YOLOv5CocoDataset',
    metainfo=dict(classes=['object']),
    data_root='/mnt/sda1/datasets/drone',
    ann_file='coco/fod_objectness.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline)

# COCO replay（data_prep/make_coco_replay.py 严格口径 31,623 图 15.9 万框，>256px 占 17.6%）：
# FOD 全是航拍小目标，纯 FOD 微调把尺度先验拉向小目标，大目标被拆成碎片框（UVSD 实测）
replay_dataset = dict(
    type='YOLOv5CocoDataset',
    metainfo=dict(classes=['object']),
    data_root='/mnt/sda1/datasets/coco',
    ann_file='annotations/coco_replay_objectness.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline)

coco_train_dataset = dict(
    _delete_=True,
    type='ConcatDataset',
    datasets=[fod_dataset, replay_dataset])

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=coco_train_dataset)

# 不做 COCO val（域无关测不出 objectness 域适配），效果用 SkyFind 验证图实测大框比例
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None
