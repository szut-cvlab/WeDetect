#!/usr/bin/env python3
"""VLM 按框坐标命名（WeDetect 数据引擎口径，一框一调用）：cod_uni_boxes_fodcoco_sam3.jsonl
→ Qwen3.8 逐框送审出 coarse/mid/fine 三段类名；框内没有以它为主体的单一物体 → 拒识剔除。
送图：裁剪窗 3 倍框（最小 128px）、小于 448px 上采样到 448，目标用品红
外扩 3px 描边标注（WeDetect 数据引擎同款视觉锚点思路）；坐标用发送图像的绝对像素
（Qwen2.5-VL 起官方约定，非 Qwen2-VL 的 0-1000 归一化）。
fine 不确定回填 mid；贴边截断但可辨认的正常标注。
输入  /mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3.jsonl
输出  /mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3_named.jsonl，一行一框：
      {file, i, box: [x,y,w,h], score, coarse, mid, fine}；拒识行 {file, i, box, score, reject: true}
      全部落盘：拒识也是决策结果，续跑不重问；下游取框时过滤带 reject 的行
断点  输出文件已含的 (file, i) 跳过（含拒识行），续跑只补缺失的框；LIMIT=N 冒烟
重问  REASK=1 运行：先剔除输出中全部拒识行再跑（提示词升级 v2 后重问 110k 拒识框用）
配置  CONCURRENCY=14 发送线程、PREP_THREADS=4 预处理线程（两段流水线），均为文件内常量
运行  .venv/bin/python data_prep/vlm_name_boxes.py                     # 全量
      .venv/bin/python data_prep/vlm_name_boxes.py <file> [box_idx]    # 单框试跑，如 000811.jpg 5
"""
import base64
import io
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import requests
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 池里有截断坏图，与 skyfind_dedup.py 同款处理

REPO = Path(__file__).resolve().parent.parent

# BASE_URL = 'https://api.siliconflow.cn/v1'
# API_KEY = 'sk-jeyxukffrtwzwxvyenvsedlepcviaafdkjeihqiheuuhncat'
# MODEL = 'Qwen/Qwen3.8-27B'
# BASE_URL = 'https://api-inference.modelscope.cn/v1'
# API_KEY = 'ms-d95e068b-f191-4e5a-8f35-62a137644619'
# MODEL = 'Qwen/Qwen3.8-Flash-Next'

BASE_URL = 'http://10.18.32.108:30000/v1'
API_KEY = ''
MODEL = '/data1/models/hf/Qwen3.8-27B-FP8'

JSONL_IN = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3.jsonl')
JSONL_OUT = Path('/mnt/sda1/datasets/drone/SkyFind/cod_uni_boxes_fodcoco_sam3_named.jsonl')
ROOT = Path('/mnt/sda1/datasets/drone/SkyFind/train')  # 图片目录

RETRY = 3        # 指数退避
CONCURRENCY = 14    # 发送线程数（只做网络 IO）
PREP_THREADS = 4    # 预处理线程数（读图/裁剪/编码，与发送并行）
BATCH = 32          # 预处理批次大小，控内存

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=CONCURRENCY,
                                         pool_maxsize=CONCURRENCY * 2)
_session.mount('http://', _adapter)
_session.mount('https://', _adapter)

PROMPT_HEAD = '''You label object boxes for drone aerial images. One target box is specified by absolute pixel coordinates (x1,y1,x2,y2) of the image and outlined by a magenta rectangle. Decide what single object the box frames and output three granularity levels:
coarse: broad category word (e.g. vehicle, person, boat, building, animal)
mid: common object type (e.g. car, van, truck, bus, motor, bicycle, pedestrian, boat, ship, house)
fine: specific variant word (e.g. sedan, hatchback, SUV, pickup, taxi, dump truck, sailboat)
Rules:
- The box is the outer bounding box of ONE object. In aerial views neighboring objects often touch or partially intrude into the box edges; this is normal and NOT a group. Label the object that fills most of the box area.
- Judge only what is inside the box. Output {"reject": true} only if no single object fills most of the box area: empty ground/road/water/roof, or the box falls in a gap between objects.
- If the fine variant is uncertain, repeat the mid word as fine.
- Truncated at image edge but recognizable: keep and label normally.
Output ONLY one JSON object, e.g. {"coarse":"vehicle","mid":"car","fine":"sedan"} or {"reject":true}'''


def encode_img(img):
    """PIL 图 → JPEG b64"""
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm(img_b64, box):
    """单框送审 → ((coarse, mid, fine) or None, 原始输出文本)，失败抛异常"""
    r = _session.post(
        f'{BASE_URL}/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}'},
        json={'model': MODEL, 'temperature': 0, 
            "chat_template_kwargs": {"enable_thinking": False},
            'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                {'type': 'text', 'text': f'{PROMPT_HEAD}\nBox: ({box[0]}, {box[1]}, {box[2]}, {box[3]})'},
            ]}]}, timeout=300)
    r.raise_for_status()
    raw = r.json()['choices'][0]['message']['content']
    text = raw[raw.find('{'):raw.rfind('}') + 1]
    item = json.loads(text)
    if item.get('reject'):
        return None, raw
    c, m, f = (str(item.get(k, '') or '').strip().lower() for k in ('coarse', 'mid', 'fine'))
    m = m or c
    f = f or m
    if not m:
        return None, raw
    return (c or m, m, f), raw


def crop_xyxy(box, W, H):
    """裁剪窗：3 倍框尺寸（每侧扩 1 倍框）、最小 128px——窗口随框走，小框在窗口里占比不至于被淹没；clamp 到图内"""
    x1, y1, x2, y2 = box
    pad_x = max((x2 - x1), (128 - (x2 - x1)) / 2)
    pad_y = max((y2 - y1), (128 - (y2 - y1)) / 2)
    return (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
            min(W, int(x2 + pad_x)), min(H, int(y2 + pad_y)))


def make_crop(img, box, W, H):
    """裁剪窗（3 倍框、最小 128px）+ 小窗上采样到 448 + 品红外扩描边标目标 → (crop, box_abs, window)"""
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_xyxy(box, W, H)
    crop = img.crop((cx1, cy1, cx2, cy2))
    w, h = crop.size
    s = max(1.0, 448 / max(w, h))
    if s > 1:
        crop = crop.resize((round(w * s), round(h * s)), Image.LANCZOS)
    box_abs = (round((x1 - cx1) * s), round((y1 - cy1) * s),
               round((x2 - cx1) * s), round((y2 - cy1) * s))
    # 品红外扩 3px 描边：零像素覆盖目标，且品红在航拍场景中不存在、不会被误读为目标颜色
    ImageDraw.Draw(crop).rectangle([box_abs[0] - 3, box_abs[1] - 3, box_abs[2] + 3, box_abs[3] + 3],
                                   outline='magenta', width=3)
    return crop, box_abs, (cx1, cy1, cx2, cy2)


def prep_image(rec, done):
    """单图预处理（CPU 密集）：读图 + 逐框裁剪/上采样/编码 → payload 列表，跳过断点命中框"""
    file = rec['file']
    img = Image.open(ROOT / file).convert('RGB')
    W, H = rec['width'], rec['height']
    payloads = []
    for i, b in enumerate(rec['boxes']):
        if (file, i) in done:
            continue  # 断点命中，不重复请求
        x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        crop, box_abs, _ = make_crop(img, (x1, y1, x2, y2), W, H)
        payloads.append((file, i, b[:4], b[4], encode_img(crop), box_abs))
    return payloads


def send_box(fout, lock, payload, stats):
    """发送单框（纯网络 IO，重试退避），成功即写一行（拒识不写）"""
    file, i, box, score, img_b64, box_abs = payload
    for attempt in range(RETRY):
        try:
            named, _ = call_vlm(img_b64, box_abs)
            break
        except Exception:
            if attempt == RETRY - 1:
                raise
            time.sleep(10 * 2 ** attempt)  # 10/20/40/80/160s 指数退避
    if named is None:
        with lock:
            stats['reject'] += 1
            fout.write(json.dumps({'file': file, 'i': i, 'box': box, 'score': score,
                                   'reject': True}, ensure_ascii=False) + '\n')
            fout.flush()
        return  # 拒识也落盘，续跑不重问，下游过滤 reject 行
    with lock:
        stats['ok'] += 1
        fout.write(json.dumps({'file': file, 'i': i, 'box': box, 'score': score,
                               'coarse': named[0], 'mid': named[1], 'fine': named[2]},
                              ensure_ascii=False) + '\n')
        fout.flush()


def strip_rejects():
    """REASK 前置：剔除输出文件中全部拒识行（原子替换），这些框回到待跑队列，按当前提示词重问"""
    kept, n = [], 0
    for line in open(JSONL_OUT):
        try:
            if json.loads(line).get('reject'):
                n += 1
                continue
        except Exception:
            pass
        kept.append(line if line.endswith('\n') else line + '\n')
    tmp = JSONL_OUT.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        f.writelines(kept)
    os.replace(tmp, JSONL_OUT)
    return n


def load_records():
    """读输入 jsonl + 输出文件的 (file, i) 断点；返回 (recs, done)"""
    done = set()
    if JSONL_OUT.exists():
        for line in open(JSONL_OUT):
            try:
                r = json.loads(line)
                done.add((r['file'], r['i']))
            except Exception:
                pass
    recs = [json.loads(line) for line in open(JSONL_IN)]
    return recs, done


def run_full():
    """跑全量：两段流水线——预处理线程池产 payload（界队列缓冲），发送线程池纯网络 IO，框级断点续跑"""
    recs, done = load_records()
    limit = int(os.environ.get('LIMIT', 0))
    if limit:
        recs = recs[:limit]
    todo = [r for r in recs if 'boxes' in r and r['boxes']
            and any((r['file'], i) not in done for i in range(len(r['boxes'])))]
    n_cached = sum(1 for r in recs if 'boxes' in r
                   for i in range(len(r['boxes'])) if (r['file'], i) in done)
    print(f'共 {len(recs)} 张，待跑 {len(todo)} 张（断点命中 {n_cached} 框），模型 {MODEL}，'
          f'发送并发 {CONCURRENCY}，预处理并发 {PREP_THREADS}', flush=True)
    if not todo:
        return

    q = queue.Queue(maxsize=256)  # 逐框队列，发送层按框调度（解码开销已在预处理层按图摊掉）
    stats = {'ok': 0, 'reject': 0, 'fail': 0, 'prep_fail': 0}
    n_pending = sum(1 for r in todo for i in range(len(r['boxes'])) if (r['file'], i) not in done)
    pbar = tqdm(total=n_pending, desc='vlm naming', unit='box')
    _lock = threading.Lock()

    def prep_safe(rec):
        try:
            return prep_image(rec, done)
        except Exception as e:
            with _lock:
                stats['prep_fail'] += 1
            print(f'[prep 失败] {rec["file"]}: {e}', flush=True)
            return None  # 坏图跳过，不终止全量

    def producer():
        try:
            with ThreadPoolExecutor(PREP_THREADS) as pp:
                it = iter(todo)
                batch = list(islice(it, BATCH))
                while batch:
                    for fut in as_completed({pp.submit(prep_safe, r): r for r in batch}):
                        for p in fut.result() or []:
                            q.put(p)  # 谁先完成谁先入队，逐框供给
                    batch = list(islice(it, BATCH))
        finally:
            q.put(None)

    def consume():
        while True:
            p = q.get()
            if p is None:
                q.put(None)  # 结束信号传递给其余 consumer
                pbar.close()
                return
            try:
                send_box(fout, _lock, p, stats)
            except Exception:
                with _lock:
                    stats['fail'] += 1  # 失败框不落盘，下次续跑补
            pbar.update(1)

    threading.Thread(target=producer, daemon=True).start()
    with open(JSONL_OUT, 'a') as fout, ThreadPoolExecutor(CONCURRENCY) as pool:
        futs = [pool.submit(consume) for _ in range(CONCURRENCY)]
        for fut in as_completed(futs):
            fut.result()
    pbar.close()
    print(f"完成：成功 {stats['ok']}，拒识 {stats['reject']}（已落盘），"
          f"发送失败 {stats['fail']}，预处理失败 {stats['prep_fail']}（不落盘，续跑自动补）", flush=True)


def test_one(file='000811.jpg', box_idx=0):
    """单框试跑：跑第 box_idx 个框，存带红框的裁剪图/整图（工作区根目录），打印 VLM 原始输出与解析结果。
    送审用无描边裁剪图（与全量口径一致），红框图仅供人工核对位置。"""
    rec = None
    for line in open(JSONL_IN):
        r = json.loads(line)
        if r['file'] == file:
            rec = r
            break
    if rec is None:
        print(f'{file} 不在 {JSONL_IN.name} 里')
        return
    if 'boxes' not in rec or not rec['boxes']:
        print(f'{rec}')
        return
    if not 0 <= box_idx < len(rec['boxes']):
        print(f'box_idx {box_idx} 越界，共 {len(rec["boxes"])} 框')
        return

    stem = Path(file).stem
    b = rec['boxes'][box_idx]
    x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    W, H = rec['width'], rec['height']
    img = Image.open(ROOT / file).convert('RGB')
    crop, box_abs, window = make_crop(img, (x1, y1, x2, y2), W, H)

    crop.save(REPO / f'vlm_test_{stem}_crop.jpg', quality=90)  # 与送审图一致（含品红描边）
    full = img.copy()
    ImageDraw.Draw(full).rectangle([x1, y1, x2, y2], outline='red', width=max(2, min(img.size) // 1000))
    full.resize((1600, int(1600 * H / W))).save(REPO / f'vlm_test_{stem}_full.jpg', quality=85)

    img_b64 = encode_img(crop)
    t0 = time.time()
    named, raw = call_vlm(img_b64, box_abs)
    print(f'框 {box_idx}/{len(rec["boxes"])}：原图 ({x1},{y1},{x2},{y2})，裁剪窗 {window}，'
          f'送审坐标 {box_abs}，耗时 {time.time() - t0:.1f}s')
    print(f'VLM 原始输出: {raw}')
    print(f'解析: {named}')


def vis_reject(n=8, seed=0):
    """人工抽检拒识质量：随机抽 n 个 reject 框，还原送审裁剪图（品红描边）+ 整图红框，存工作区根目录"""
    import random
    rejects = [json.loads(l) for l in open(JSONL_OUT) if '"reject": true' in l or '"reject":true' in l]
    picks = random.Random(seed).sample(rejects, n)
    by_file = {}
    for p in picks:
        by_file.setdefault(p['file'], []).append(p)
    for file, ps in by_file.items():
        rec = next(json.loads(l) for l in open(JSONL_IN) if json.loads(l)['file'] == file)
        img = Image.open(ROOT / file).convert('RGB')
        W, H = rec['width'], rec['height']
        stem = Path(file).stem
        for p in ps:
            b = p['box']
            x1, y1, x2, y2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            crop, _, _ = make_crop(img, (x1, y1, x2, y2), W, H)
            crop.save(REPO / f'vlm_reject_{stem}_i{p["i"]}_crop.jpg', quality=90)
            full = img.copy()
            ImageDraw.Draw(full).rectangle([x1, y1, x2, y2], outline='red', width=max(2, min(img.size) // 1000))
            full.resize((1600, int(1600 * H / W))).save(REPO / f'vlm_reject_{stem}_i{p["i"]}_full.jpg', quality=85)
            print(f'{file} i={p["i"]} box=({x1},{y1},{x2},{y2}) score={p["score"]:.2f} '
                  f'→ vlm_reject_{stem}_i{p["i"]}_crop.jpg / _full.jpg')


if __name__ == '__main__':
    if os.environ.get('REASK'):
        print(f'剔除拒识行 {strip_rejects()} 条，按当前提示词重问', flush=True)
    # test_one()
    # vis_reject(8)
    run_full()
