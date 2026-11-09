#!/usr/bin/env python
"""Pexels 抓取：search 只写 CSV（带 mp4 直链与检索词），download 并发下 mp4。

用法    python3 data_prep/pexels.py search "container terminal" [每页=80] [--pages 3]
        python3 data_prep/pexels.py filter [--apply]
        python3 data_prep/pexels.py sheets          # 待判拼图 → /tmp/pending.jpg
        python3 data_prep/pexels.py download [--limit N] [--jobs 8]
产物    crawl/pexels/videos.csv（视频ID,作者,分辨率,时长s,MB,url,帧,文件）
        crawl/pexels/videos/<视频ID>.mp4
"""
import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from tqdm import tqdm

OUT = "/mnt/sda1/datasets/drone/crawl/pexels"
CSV = f"{OUT}/videos.csv"


def key():
    load_dotenv("/home/liuwei/projects/research/WeDetect/.env")
    return os.environ.get("PEXELS_API_KEY") or sys.exit("缺 PEXELS_API_KEY")


def get(url, k):
    h = {"Authorization": k, "User-Agent": "Mozilla/5.0"}   # 缺 UA 会被 Cloudflare 拦（1010）
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60) as r:
        return json.loads(r.read())


def cmd_search(a):
    k = key()
    seen = {r["视频ID"] for r in csv.DictReader(open(CSV, encoding="utf-8"))} if os.path.exists(CSV) else set()
    new = not os.path.exists(CSV)
    res = []
    for page in range(1, a.pages + 1):
        try:
            res += get("https://api.pexels.com/videos/search?"
                       + urllib.parse.urlencode({"query": a.query, "per_page": min(80, a.num), "page": page}), k)["videos"]
        except Exception as e:
            print(f"  第 {page} 页失败：{type(e).__name__}")
            break
    n = 0
    with open(CSV, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["视频ID", "作者", "分辨率", "时长s", "MB", "url", "帧", "文件", "检索词"])
        for v in res:
            vid = str(v["id"])
            if vid in seen:
                continue
            files = v["video_files"]
            cand = [x for x in files if x["width"] <= 1920] or files   # 最高取 1080p
            f = max(cand, key=lambda x: (x["width"], x["height"]))
            seen.add(vid)
            n += 1
            wr.writerow([vid, v["user"]["name"], f"{f['width']}x{f['height']}", v["duration"],
                         round(f.get("size", 0) / 1e6, 1), v["url"], v["image"], f["link"], a.query])
        fh.flush()
    print(f"新增 {n} 条 → {CSV}")


KEYS = ("drone", "aerial", "from-above", "top-view", "bird", "overhead", "fly-over", "flyover")


def cmd_filter(a):
    """结构规则筛（不下视频、不判画面）：
    排除：竖屏 h>w / 时长 <5s / 宽 <1280 / 已是文件且被人工判过
    待判：slug 里没有航拍线索（drone/aerial/top-view/bird/overhead）
    保留：slug 有线索
    """
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    hdr = list(rows[0].keys())
    if "判定" not in hdr:
        hdr.append("判定")
    stat = {}
    for r in rows:
        if r.get("判定") == "保留":
            stat["保留"] = stat.get("保留", 0) + 1      # 已人工确认的保留，不重新判
            continue
        w, h = (int(x) for x in r["分辨率"].split("x"))
        slug = r["url"].rstrip("/").rsplit("/", 1)[-1]
        if h > w:
            r["判定"] = "排除/竖屏"
        elif float(r["时长s"]) < 5:
            r["判定"] = "排除/过短"
        elif w < 1280:
            r["判定"] = "排除/低清"
        elif any(k in slug for k in KEYS):
            r["判定"] = "保留"
        else:
            r["判定"] = "待判"
        stat[r["判定"]] = stat.get(r["判定"], 0) + 1
    if a.apply:
        with open(CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr)
            w.writeheader(); w.writerows(rows)
    print("判定分布：", stat, "→", CSV if a.apply else "（未写盘，加 --apply 才写）")


def cmd_sheets(a):
    """把「待判」的中间帧拼成 5 列网格图，位置→ID 写 json。"""
    from PIL import Image
    D = OUT
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r.get("判定") == "待判"]
    pick = []
    for r in rows:
        d = f"{D}/videos/frames/{r['视频ID']}"
        if not os.path.isdir(d):
            continue
        fs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if fs:
            pick.append((r["视频ID"], f"{d}/{fs[len(fs) // 2]}"))
    if not pick:
        print("没有待判的帧")
        return
    COLS, ROWS, CW, CH = 5, 8, 300, 170
    per = COLS * ROWS
    mapping = {}
    for gi in range((len(pick) + per - 1) // per):
        batch = pick[gi * per:(gi + 1) * per]
        sheet = Image.new("RGB", (CW * COLS, CH * ROWS), "white")
        for i, (_, path) in enumerate(batch):
            im = Image.open(path).convert("RGB").resize((CW - 2, CH - 2))
            r_, c_ = divmod(i, COLS)
            sheet.paste(im, (c_ * CW + 1, r_ * CH + 1))
        k = gi + 1
        sheet.save(f"/tmp/pending_{k}.jpg", quality=78)
        mapping[str(k)] = [v for v, _ in batch]
        print(f"  /tmp/pending_{k}.jpg  {len(batch)} 条")
    json.dump(mapping, open("/tmp/pending_ids.json", "w"))
    print(f"共 {len(pick)} 条，位置键 = 图号 + (行-1)*5+(列-1)")


def fetch(row):
    p = f"{OUT}/videos/{row['视频ID']}.mp4"
    for _ in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(row["文件"], headers={"User-Agent": "Mozilla/5.0"}),
                                        timeout=300) as resp, open(p, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return row["视频ID"], True
        except Exception:
            continue
    return row["视频ID"], False


def cmd_download(a):
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["文件"].endswith(".mp4")]
    if rows and "判定" in rows[0]:
        rows = [r for r in rows if r["判定"] in ("保留", "待判")]
    rows.sort(key=lambda r: float(r["MB"] or 0))
    todo = [r for r in rows if not os.path.exists(f"{OUT}/videos/{r['视频ID']}.mp4")]
    gb = sum(float(r["MB"] or 0) for r in todo) / 1000
    if a.limit:
        todo = todo[:a.limit]
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    print(f"已完成 {len(rows) - len(todo)}/{len(rows)}，本次下 {len(todo)} 条 / {gb:.1f} GB，并发 {a.jobs}")
    fail = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(fetch, r) for r in todo]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="下载", unit="条"):
            vid, ok = fu.result()
            if not ok:
                fail += 1
                tqdm.write(f"  失败 {vid}")
    print(f"完成，失败 {fail} 条")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("num", nargs="?", type=int, default=80, help="每页条数，上限 80")
    s.add_argument("--pages", type=int, default=1)
    f = sub.add_parser("filter"); f.add_argument("--apply", action="store_true")
    sub.add_parser("sheets")
    d = sub.add_parser("download"); d.add_argument("--limit", type=int, default=0); d.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    {"search": cmd_search, "filter": cmd_filter, "sheets": cmd_sheets, "download": cmd_download}[a.cmd](a)


if __name__ == "__main__":
    main()
