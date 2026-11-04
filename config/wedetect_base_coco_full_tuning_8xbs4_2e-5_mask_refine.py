_base_ = ["wedetect_base.py"]


load_from = 'wedetect_base.pth'

num_training_classes = 80
num_classes = 80

base_lr = 2e-5
weight_decay = 0.05
train_batch_size_per_gpu = 4
max_epochs = 12  # Maximum training epochs
close_mosaic_epochs = 4
save_epoch_intervals = 1
# persistent_workers must be False if num_workers is 0
persistent_workers = True

img_scale = (640, 640)
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
mixup_prob = 0.15

model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    # backbone=dict(
    #     text_model=dict(
    #         frozen_modules=['all'],
    #     ),
    # ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_training_classes,
        ),
    ),
    train_cfg=dict(assigner=dict(num_classes=num_classes,)),
)


# dataset settings

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOLoadAnnotations', with_bbox=True, with_mask=True, mask2bbox=True)
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
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
        scaling_ratio_range=(1 - affine_scale,
                             1 + affine_scale),
        # img_scale is (width, height)
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
        prob=mixup_prob,
        pre_transform=[*pre_transform,
                       *mosaic_affine_transform]),
    dict(type='RemoveDataElement', keys=['gt_masks']),
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
    *text_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='RemoveDataElement', keys=['gt_masks']),
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
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
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
    *text_transform
]


coco_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='WeCocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_zh_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    num_workers=2,
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=coco_train_dataset)


# training settings
param_scheduler = [
    # 1. 线性预热 (Warmup)
    dict(
        type='LinearLR',
        start_factor=0.001, # 预热开始时的倍率
        end_factor=1.0,     # 预热结束时的倍率
        begin=0,
        end=1000,           # 预热持续 1000 次迭代
        by_epoch=False      # 按 iteration 计数
    ),
    # 2. 主训练过程线性递减 (Main Linear Decay)
    dict(
        type='LinearLR',
        start_factor=1.0,   # 衰减开始倍率
        end_factor=0.001,   # 衰减结束倍率
        begin=0,            # 从第 0 epoch 开始计算衰减曲线
        end=max_epochs,     # 到第 12 epoch 结束
        by_epoch=True,      # 按 epoch 规划
        convert_to_iter_based=True # 转换为 iter 模式，实现平滑下降
    )
]

custom_hooks = [
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)])

optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={'backbone.text_model': dict(lr_mult=0.01),
                     'logit_scale': dict(weight_decay=0.0)}),
    constructor='YOLOWv5OptimizerConstructor')



