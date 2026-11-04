_base_ = ["wedetect_tiny.py"]

# LVIS minival 评测（对应 base 文件底部注释掉的用法；独立配置避免改默认行为）
val_dataloader = dict(
    num_workers=8,
    dataset=_base_.lvis_minival_dataset,
)
test_dataloader = val_dataloader
val_evaluator = _base_.lvis_minival_evaluator
test_evaluator = val_evaluator
