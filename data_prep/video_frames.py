#!/usr/bin/env python
"""按秒抽帧 + dHash 去重，图写进 <视频目录>/frames/<视频名>/（不写 CSV）。

用法    python3 data_prep/video_frames.py <视频目录> [间隔秒=3] [--min-side N] [--limit N] [--jobs 4]
产物    <视频目录同级>/frames/<视频名>/000001.jpg …
"""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm


def dhash(p):
    g = Image.open(p).convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def one(args):
    src, f, out, every, min_side = args
    name = os.path.splitext(f)[0]
    v = os.path.join(src, f)
    d = os.path.join(out, name)
    if os.path.isdir(d) and any(x.endswith(".jpg") for x in os.listdir(d)):
        return name, -1, "已抽过"                      # -1 = 跳过
    try:
        wh = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "json", v],
            capture_output=True, text=True).stdout)["streams"][0]
    except Exception:
        return name, -1, "ffprobe 失败"
    if min_side and min(wh["width"], wh["height"]) < min_side:
        return name, -1, f"短边 {wh['width']}x{wh['height']}"

    os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v,
                    "-vf", f"fps=1/{every}", "-q:v", "2", f"{d}/%06d.jpg"], check=False)
    kept = []
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        try:
            h = dhash(p)
        except Exception:
            continue
        if any(bin(h ^ o).count("1") <= 10 for o in kept):
            subprocess.run(["rm", "-f", p])
            continue
        kept.append(h)
    return name, len(kept), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("every", nargs="?", type=float, default=3.0)
    ap.add_argument("--min-side", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()

    out = os.path.join(os.path.dirname(a.src.rstrip("/")), "frames")
    os.makedirs(out, exist_ok=True)
    vids = [f for f in sorted(os.listdir(a.src)) if f.lower().endswith((".mp4", ".mkv", ".webm", ".flv", ".mov"))]
    if a.limit:
        vids = vids[:a.limit]

    total = skip = 0
    reasons = []
    bar = tqdm(total=len(vids), desc="抽帧", unit="条")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(one, (a.src, f, out, a.every, a.min_side)) for f in vids]
        for fu in as_completed(futs):
            name, n, why = fu.result()
            if n < 0:
                skip += 1
                if why != "已抽过":
                    reasons.append(f"{name}: {why}")
            else:
                total += n
            bar.update(1)
            bar.set_postfix(帧=total, 跳过=skip)
    bar.close()
    print(f"完成：{len(vids)} 条，保留 {total} 帧，跳过 {skip} 条 → {out}")
    for r in reasons:
        print("  跳过", r)


if __name__ == "__main__":
    main()
