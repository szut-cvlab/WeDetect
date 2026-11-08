#!/usr/bin/env python
"""类名写法检查：XLM-R 怎么切分下划线/连字符，以及不同写法在文本塔里的距离。

用途是给「类名只做分隔符归一、不改写词义」这条规则提供数据：
  1) 官方名 freight_car / awning-tricycle 与空格版的余弦（噪声级差异）
  2) 改写版（tricycle with canopy）与空格版的余弦（差别也不大，不值得偏离官方名）
  3) 修饰词是否把语义拉向中心词（freight car→truck、fishing boat→boat）

推理路径与训练一致：文本塔权重取自 checkpoints/WeDetect/wedetect_base.pth 的
backbone.text_model.model.*（XLMRobertaModel）+ backbone.text_model.head.*（投影层），
池化取 last_hidden_state[:,0] 再过投影头并 L2 归一化（见 wedetect/models/backbones/mm_backbone.py）。

运行：.venv/bin/python data_prep/check_class_tokenizer.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, XLMRobertaModel

CKPT = "checkpoints/WeDetect/wedetect_base.pth"
NAME = "./xlm-roberta-base"

ANCHORS = ["car", "truck", "van", "bus", "boat", "tricycle", "bicycle", "motorcycle", "motor"]
VARIANTS = [  # (概念, 官方写法, 空格写法, 改写写法或 None)
    ("货运车厢", "freight_car", "freight car", None),
    ("罐车", "tank_truck", "tank truck", None),
    ("带篷三轮", "awning-tricycle", "awning tricycle", "tricycle with canopy"),
    ("三轮车(对照)", "tricycle", "tricycle", None),
]
EXTRA = ["freight", "awning", "canopy", "fishing", "tank", "cargo tricycle", "tuk-tuk", "rickshaw",
         "excavator", "crane", "SUV", "vehicle", "fishing boat"]

WORDS = ANCHORS + sorted({w for v in VARIANTS for w in v[1:] if w}) + EXTRA

tok = AutoTokenizer.from_pretrained(NAME)
model = XLMRobertaModel(AutoConfig.from_pretrained(NAME))
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
pre = "backbone.text_model.model."
model.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
head = torch.nn.Linear(768, 768)
head.load_state_dict({k.split(".")[-1]: v for k, v in sd.items()
                      if k.startswith("backbone.text_model.head.")})
model.eval()
head.eval()

print("=== 1. 分词 ===")
for w in WORDS:
    print(f"{w!r:22} -> {[tok.decode([i]) for i in tok(w)['input_ids']]}")

with torch.no_grad():
    enc = tok(WORDS, padding=True, return_tensors="pt")
    e = F.normalize(head(model(**enc)["last_hidden_state"][:, 0]), dim=-1)
sim = (e @ e.T).numpy()
idx = {w: i for i, w in enumerate(WORDS)}


def s(a, b):
    return sim[idx[a]][idx[b]]


print("\n=== 2. 写法之间的余弦（差异越小，说明归一只是卫生问题）===")
for _, off, spc, para in VARIANTS:
    line = f"  {off:16} vs {spc:16} {s(off, spc):.3f}"
    if para:
        line += f"   空格版 vs 改写版 {s(spc, para):.3f}"
    print(line)

print("\n=== 3. 各写法对锚类的相似度，取最近三个 ===")
for w in sorted(set(WORDS) - set(ANCHORS)):
    top = sorted(ANCHORS, key=lambda a: -s(w, a))[:3]
    print(f"  {w:22}" + "  ".join(f"{a} {s(w, a):.3f}" for a in top))

print("\n=== 4. 修饰词有没有把语义拉向中心词 ===")
for pair, mod, head_w, anchor in [("freight car", "freight", "car", "truck"),
                                  ("fishing boat", "fishing", "boat", "boat"),
                                  ("tank truck", "tank", "truck", "truck")]:
    print(f"  {mod:9} 单独→{anchor} {s(mod, anchor):.3f} | {head_w:9} 单独→{anchor} "
          f"{s(head_w, anchor):.3f} | 组合 {pair:14}→{anchor} {s(pair, anchor):.3f}")
