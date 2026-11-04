_base_ = ["wedetect_tiny.py"]

# 数据侧差异与 base 版完全一致，直接借过来；mmengine 不允许多 base 合并，所以只能这样引用
from config.wedetect_base_visdrone import val_dataloader, val_evaluator

test_dataloader = val_dataloader
test_evaluator = val_evaluator
