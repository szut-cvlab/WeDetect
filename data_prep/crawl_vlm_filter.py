#!/usr/bin/env python3
"""自爬池 VLM 航拍过滤：读 cod.json 的数字命名图（自爬），逐张走 OpenAI 兼容 API 判航拍/地拍，
结果流式追加到 /mnt/sda1/datasets/drone/SkyFind/crawl_labels.json，每行 {"file","label","note"}。
断点续跑：已判过的 file 自动跳过；失败的不写文件，下次跑自动重试。

配置（环境变量）：
    VLM_BASE_URL  API 地址，如 https://api.xx.com/v1
    VLM_API_KEY   密钥
    VLM_MODEL     模型名，如 qwen-vl-max
用法：
    python data_prep/crawl_vlm_filter.py            # 全量
    LIMIT=30 python data_prep/crawl_vlm_filter.py   # 试跑前 30 张（锚点对拍用）
"""
import base64
import io
import json
import os
import re
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

COD_JSON = Path('/mnt/sda1/datasets/drone/SkyFind/cod.json')
OUT_JSONL = Path('/mnt/sda1/datasets/drone/SkyFind/crawl_labels.json')
MAX_EDGE = 1024     # 送审前压边，省带宽
RETRY = 3

BASE_URL = 'https://api.siliconflow.cn/v1'
API_KEY = 'sk-jeyxukffrtwzwxvyenvsedlepcviaafdkjeihqiheuuhncat'
MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'

PROMPT = '''判断这张照片的拍摄视角。输出 JSON：{"label": "aerial 或 ground", "note": "一句话画面描述"}。

判定标准（已由人工标定 20 张校准）：
- aerial：任何空中视角，包括垂直俯拍、斜视角航拍、无人机低空跟拍
- ground：明确的地面视角，即相机在地面/街道高度拍摄（街景、公园、车厢内外平视等）
- 边界不确定时判 aerial：只有明确是地面拍摄才判 ground
- note 用中文简短描述画面内容，不超过 15 字

只输出 JSON，不要其他文字。'''


def encode_img(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        r = MAX_EDGE / max(w, h)
        img = img.resize((round(w * r), round(h * r)))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def judge(rel, img_b64):
    """单图送审 → (label, note)，失败抛异常"""
    r = requests.post(
        f'{BASE_URL}/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={
            'model': MODEL,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'image_url',
                     'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                    {'type': 'text', 'text': PROMPT},
                ],
            }],
            'temperature': 0,
        }, timeout=120)
    r.raise_for_status()
    text = r.json()['choices'][0]['message']['content'].strip()
    text = text[text.find('{'):text.rfind('}') + 1]  # 剥掉可能包着的 ```json 围栏
    d = json.loads(text)
    label = d.get('label', '')
    assert label in ('aerial', 'ground'), f'非法 label: {label!r}'
    return label, str(d.get('note', ''))[:40]


def main():
    cod = json.load(open(COD_JSON))
    root = Path(cod['root'])
    todo = [f for f in cod['images'] if re.match(r'^\d{6}\.jpg$', f)]

    done = set()
    if OUT_JSONL.exists():
        for line in open(OUT_JSONL):
            try:
                done.add(json.loads(line)['file'])
            except (json.JSONDecodeError, KeyError):
                continue
    limit = int(os.environ.get('LIMIT', '0'))
    if limit:
        todo = todo[:limit]
    todo = [f for f in todo if f not in done]
    print(f'自爬共 {len(done) + len(todo)} 张，已判 {len(done)}，本趟 {len(todo)}，输出 {OUT_JSONL}')

    n_err = 0
    with open(OUT_JSONL, 'a', buffering=1) as fout, tqdm(todo, desc='航拍过滤', unit='图') as bar:
        for rel in bar:
            for i in range(RETRY):
                try:
                    label, note = judge(rel, encode_img(root / rel))
                    fout.write(json.dumps({'file': rel, 'label': label, 'note': note},
                                          ensure_ascii=False) + '\n')
                    break
                except Exception as e:
                    if i == RETRY - 1:
                        n_err += 1
                        bar.set_postfix(err=n_err, last=f'{type(e).__name__}')
            else:
                continue  # 重试耗尽，不写文件，下次自动重试

    print(f'完成：本趟 {len(todo)}，失败 {n_err}（下次重跑自动补），输出 {OUT_JSONL}')


if __name__ == '__main__':
    main()
