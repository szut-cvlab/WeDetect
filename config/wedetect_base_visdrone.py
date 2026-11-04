_base_ = ["wedetect_base.py"]

from eval_visdrone.build_coco import CATS   # 类名与 GT json 的 categories 同源，不会漂移

DATA_ROOT = "data/VisDrone/"
ANN = "annotations/visdrone_testdev_coco.json"

# 只覆盖与 COCO 不同的叶子；test_pipeline / batch / sampler 等全部继承 wedetect_base.py
val_dataloader = dict(
    dataset=dict(
        dataset=dict(data_root=DATA_ROOT,
                     ann_file=ANN,
                     data_prefix=dict(img="images/"),
                     metainfo=dict(classes=CATS)),
        class_text_path="data/texts/visdrone_zh_class_texts.json",
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=DATA_ROOT + ANN)
test_evaluator = val_evaluator
