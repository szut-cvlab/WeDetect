#!/usr/bin/env python
"""B 站抓取：search 只写 CSV（带标签与判定），download 按判定下 mp4。

用法    BILI_COOKIE_FILE=xx python3 data_prep/bili.py search "港口 航拍" [条数=20]
        BILI_COOKIE_FILE=xx python3 data_prep/bili.py download [--limit N] [--jobs 4]
产物    crawl/bili/videos.csv（bvid,cid,标题,作者,时长s,质量,url,发布时间,标签,分区,判定）
        crawl/bili/videos/<bvid>.mp4
"""
import argparse
import csv
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

OUT = "/mnt/sda1/datasets/drone/crawl/bili"
CSV = f"{OUT}/videos.csv"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36",
      "Referer": "https://www.bilibili.com/"}
MIXIN = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
         33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
         61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
         36, 20, 34, 44, 52]
POS = ("航拍", "无人机", "fpv", "飞越", "俯瞰", "上帝视角", "空中", "鸟瞰", "穿越机", "dji", "大疆")
NEG = ("科普", "教程", "纪录片", "新闻", "事故", "施工工艺", "建模", "游戏", "素材", "合集",
       "讲解", "分析", "评测", "开箱", "动画")


def cookie():
    path = os.environ.get("BILI_COOKIE_FILE", "")
    if os.path.exists(path):
        rows = [l.rstrip("\n").split("\t") for l in open(path, encoding="utf-8")]
        return "; ".join(f"{r[5]}={r[6]}" for r in rows if len(r) >= 7 and not r[0].startswith("#"))
    sd = os.environ.get("BILI_SESSDATA")
    return f"SESSDATA={sd}" if sd else None


def get(url, ck):
    h = dict(UA, **({"Cookie": ck} if ck else {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
        return json.loads(r.read())


def judge(title, tags):
    text = (title + " " + " ".join(tags)).lower()
    if any(k in text for k in POS):
        return "明确航拍"
    if any(k in text for k in NEG):
        return "排除"
    return "未知"


def cmd_search(a):
    ck = cookie()
    w = get("https://api.bilibili.com/x/web-interface/nav", ck)["data"]["wbi_img"]
    raw = w["img_url"].rsplit("/", 1)[-1][:-4] + w["sub_url"].rsplit("/", 1)[-1][:-4]
    q = {"search_type": "video", "keyword": a.kw, "page": 1, "wts": int(time.time())}
    q["w_rid"] = hashlib.md5((urllib.parse.urlencode(sorted(q.items())) + "".join(raw[i] for i in MIXIN)[:32]).encode()).hexdigest()
    res = get("https://api.bilibili.com/x/web-interface/wbi/search/type?"
              + urllib.parse.urlencode(sorted(q.items())), ck)["data"]["result"]

    seen = {r["bvid"] for r in csv.DictReader(open(CSV, encoding="utf-8"))} if os.path.exists(CSV) else set()
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["bvid", "cid", "标题", "作者", "时长s", "质量", "url", "发布时间", "标签", "分区", "判定"])
        n = 0
        for v in res:
            if n >= a.num:
                break
            bv = v["bvid"]
            if bv in seen:
                continue
            try:
                info = get(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}", ck)["data"]
                tags = [t["tag_name"] for t in get(
                    f"https://api.bilibili.com/x/tag/archive/tags?bvid={bv}", ck)["data"]]
            except Exception:
                continue
            seen.add(bv)
            n += 1
            wr.writerow([bv, info["cid"], info["title"], info["owner"]["name"], info["duration"], "-",
                         f"https://www.bilibili.com/video/{bv}",
                         time.strftime("%Y-%m-%d", time.localtime(info["pubdate"])),
                         "|".join(tags), info.get("tname", ""), judge(info["title"], tags)])
        fh.flush()
    print(f"新增 {n} 条 → {CSV}")


def fetch(row, ck):
    bv = row["bvid"]
    p = f"{OUT}/videos/{bv}.mp4"
    try:
        info = get(f"https://api.bilibili.com/x/web-interface/view?bvid={bv}", ck)["data"]
        d = get(f"https://api.bilibili.com/x/player/playurl?bvid={bv}&cid={info['cid']}&fnval=16&qn=80", ck)["data"]
        vids = d.get("dash", {}).get("video") or []
        best = max(vids, key=lambda x: x["height"]) if vids else None
        url = best["baseUrl"] if best else d["durl"][0]["url"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as resp, open(p, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return bv, True
    except Exception:
        return bv, False


def cmd_download(a):
    ck = cookie()
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["判定"] == "明确航拍"]
    todo = [r for r in rows if not os.path.exists(f"{OUT}/videos/{r['bvid']}.mp4")]
    if a.limit:
        todo = todo[:a.limit]
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    print(f"已完成 {len(rows) - len(todo)}/{len(rows)}，本次下 {len(todo)} 条")
    fail = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(fetch, r, ck) for r in todo]
        for fu in tqdm(as_completed(futs), total=len(futs), desc="下载", unit="条"):
            bv, ok = fu.result()
            if not ok:
                fail += 1
                tqdm.write(f"  失败 {bv}")
    print(f"完成，失败 {fail} 条")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("kw"); s.add_argument("num", nargs="?", type=int, default=20)
    d = sub.add_parser("download"); d.add_argument("--limit", type=int, default=0); d.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    cmd_search(a) if a.cmd == "search" else cmd_download(a)


if __name__ == "__main__":
    main()
