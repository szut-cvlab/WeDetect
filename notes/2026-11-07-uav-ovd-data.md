# UAV-OVD 数据准备

## 1. 数据下载方式

### 1.1 内网机器互传

```bash
#!/usr/bin/env bash
SRC=liuwei@10.18.32.108
cd /mnt/sda1/datasets/drone || exit 1
exec >>scp.log 2>&1

# 建复用连接，整轮只输一次密码
ssh -o ControlMaster=auto -o ControlPath=/tmp/wd -o ControlPersist=10m -N -f "$SRC"

PAR=8   # 实测：1路 3.8MB/s、2路 7.3、4路 14.8、8路 40。延迟型负载（33万个小文件），并发换吞吐，8 路最快

run() {
  local d=$1
  ssh -o ControlPath=/tmp/wd "$SRC" \
    "tar -C /data1/datasets/drone -cf - --checkpoint=100000 --checkpoint-action=echo='[$d] %u rec' $d" \
    | tar -xf -
  echo "[$d] 完成 $(date +%H:%M:%S)"
}

i=0
for d in UAVDT DroneVehicle AU-AIR HazyDet UAVid AeroScapes Semantic-Drone UDD; do
  run "$d" &
  i=$((i+1))
  (( i % PAR == 0 )) && wait -n
done
wait
echo "全部完成 $(date '+%F %T')"
```

### 1.2 百度网盘下载

安装 BaiduPCS-Go：

```bash
cd /tmp
wget "https://ghfast.top/https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/v3.9.5/BaiduPCS-Go-v3.9.5-linux-amd64.zip"
unzip -o BaiduPCS-Go-v3.9.5-linux-amd64.zip
chmod +x BaiduPCS-Go-v3.9.5-linux-amd64/BaiduPCS-Go
mv BaiduPCS-Go-v3.9.5-linux-amd64/BaiduPCS-Go /usr/local/bin/
```

浏览器登录 pan.baidu.com，F12 → Application → Storage → Cookies → `https://pan.baidu.com`，取 `BDUSS` 与 `STOKEN` 两个值。

```bash
BaiduPCS-Go login -bduss="粘贴BDUSS" -stoken="粘贴STOKEN"
BaiduPCS-Go who
```

下载：

```bash
BaiduPCS-Go config set -max_parallel 20
BaiduPCS-Go download /apps/bypy/train.zip -save /mnt/sda1/datasets/drone -p 20
```

## 2. 数据统计

### 2.1 FOD 训练

#### 2.1.0 总览

入池五集的来源：

| 数据集 | 地址 |
| --- | --- |
| VisDrone | https://github.com/VisDrone/VisDrone-Dataset |
| DroneVehicle | https://github.com/VisDrone/DroneVehicle |
| HazyDet | https://github.com/GrokCV/HazyDet |
| CODrone | https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV ，镜像 https://hf-mirror.com/datasets/huseyincavus/codrone （`codrone.zip` 7.25 GB） |
| ATR-UMOD | https://github.com/supercpy/ATR-UMOD （README 给百度网盘链接与提取码） |

入池五个数据集的产物全在 `data/coco_anno/`（软链到 `/mnt/sda1/datasets/drone/coco`），`data_root=/mnt/sda1/datasets/drone`：

| 数据集 | 入池 json | 类数 | 图 / 框（切块后） | 转换脚本 |
| --- | --- | --- | --- | --- |
| VisDrone | `visdrone_trainval.json` | 10 | 7,019 / 381,963 | `visdrone2coco.py` |
| DroneVehicle | `dronevehicle.json` | 5 | 27,654 / 452,568 | `dronevehicle2coco.py` |
| HazyDet | `hazydet_tiles.json` | 3 | 27,399 / 445,247 | `hazydet2coco.py` → `slice_coco.py` |
| CODrone | `codrone_tiles.json` | 12 | 43,440 / 532,613 | `codrone2coco.py` → `slice_coco.py` |
| ATR-UMOD | `atrumod.json` | 11 | 13,245 / 161,787 | `atrumod2coco.py` |
| **合计** | 5 份，258 MB | 21 | **118,757 / 1,974,178** | — |

合计的 **1,974,178 是训练实例数**（dataloader 真正看到的条数；CODrone 与 HazyDet 重叠带里同一目标写两份都算），对外发布与数据集命名都用它。另两个口径：源图（切块前）69,465 张，实例去重后 1,820,029 个框。类数相加 41、合并去重 21。

命名：`{dataset}.json` 是未切块标注（不切块的集它就是入池标注），`{dataset}_tiles.json` 是切块后入池标注，`{dataset}_class_texts.json` 是类名表。`hazydet.json`、`codrone.json` 只是切块输入，不进训练。

**格式与路径**：只吃标准 COCO（`WeCocoDataset`→`YOLOv5CocoDataset`→`mmdet.CocoDataset`），框一律 `[x, y, w, h]`；`file_name` 相对 `/mnt/sda1/datasets/drone` 且带集名前缀，全池共用一个 `data_root`。

**类名规则**（词表在这张表里定，各数据集小节不再逐个评论）：只用空格分隔，禁连字符与下划线；同一概念在不同源必须写成同一个词，否则合并词表时裂成两个类、两个 embedding；错拼由各自 ALIAS 收敛。

| 源写法 | 池内写法 | 出处 |
| --- | --- | --- |
| `traffic-sign`、`traffic-light` | `traffic sign`、`traffic light` | CODrone |
| `awning-tricycle` | `awning tricycle` | VisDrone |
| `freight_car` | `freight car` | DroneVehicle、ATR-UMOD |
| `tank_truck` | `tank truck` | ATR-UMOD |
| `motorcycle` | `motor` | ATR-UMOD 归并到多数派（VisDrone 34,533、CODrone 89,261 都叫 motor） |
| `suv` | `SUV` | ATR-UMOD，缩写按论文写法；两种写法 embedding 余弦 0.980，无精度差 |

合并后 21 词，无连字符与下划线残留。**待词表汇总那步处理**：`pedestrian` 与 `people`（VisDrone 内两类，语义有别但重叠大）、`boat` 与 `ship`（CODrone 内两类，434 / 7,143 框）——都是源自己的划分，不在这里替它决定。

| 原始形态 | 数据集 | 转换 |
| --- | --- | --- |
| 水平框 | VisDrone、DroneVehicle、HazyDet | 坐标直映射 |
| 旋转框 | CODrone、ATR-UMOD | 四顶点 min/max 取外接矩形，再切块 |
| 自由文本 | COD 管线 | 取 `fine`（缺则 `mid`）当类名，三段另存 |

**校验**：框数守恒、坐标不越界、每集抽 20 张画框核对、`file_name 校验 missing=0`。

**判重**：只看框不看图。两图框按 IoU ≥ 0.5 归并，`并集 ÷ 较大那张` ≤1.05 才算真冗余（留标注全的那份），≥1.2 是同一场景不同时刻（都留）。视频抽帧类先按时间抽帧再判。独立目标数另算：逐序列按帧序走，同类 IoU ≥ 0.5 就接上、30 帧未见即闭合，数轨迹条数——断裂只会多算，所以是上限。UAVDT 这么量出来 train 42.3 万框只剩 ≤1,911 个目标（平均每目标被标 221 帧），整集排除。

#### 2.1.1 VisDrone

train+val 全进池，合成一个 COCO JSON。

| 项目 | 统计 |
| --- | --- |
| 图像数 | 7,019（train 6,471 + val 548） |
| 标注行 | 393,719（空框 3） |
| annotations | 381,963（类别 1–10） |
| ignored / others | 10,189（类 0）/ 1,564（类 11），都不进 annotations |
| 逐类框数 | car 158,930、pedestrian 88,181、motor 34,533、people 32,184、van 26,931、truck 13,625、bicycle 11,767、bus 6,177、tricycle 5,857、awning tricycle 3,778 |
| 截断 / 遮挡 | 截断 14,845（3.9%）；遮挡 无 184,113 / 部分 161,112 / 严重 36,738（9.6%） |
| 目标边长 | <8px 39,511（10.3%）、8–16 103,475（27.1%）、16–32 128,931（33.8%）、32–64 77,352（20.3%）、≥64 32,694（8.6%） |
| 分辨率 | 长边最大 2000，≤2560 不切块；前六档 1400×1050（2,498）、1400×788（1,299）、1360×765（1,151）、2000×1500（772）、1916×1078（537）、960×540（371） |

表内按原图像素统计。

#### 2.1.2 DroneVehicle

datasetninja 重打包版，RGB 与红外成对，只用 RGB 半边。XML 标注两种几何并存（polygon 八点 OBB 与 bndbox 水平框，每个对象只带一种），转换都要吃；785 张空标注图疑漏标，跳过。

| 项目 | 统计（实测） |
| --- | --- |
| 标注文件 / 入池 | 28,439 对 → 27,654 图（跳过空标注） |
| 有效框 | 452,568（polygon 421,399 + bndbox 31,171） |
| 逐类框数 | car 389,777（86.1%）、truck 22,123、bus 15,333、freight car 13,400、van 11,935 |
| 每图框数 | 中位 12 / 均值 16.4 / max 206 |
| 目标边长 | 16–32px 47.6%、32–64px 45.0%、≥64px 6.2%、<16px 仅 1.2% |
| 分辨率 | 840×712，全部一致 |

**不去重**：pHash 判掉的不是同一场景——被删图 80.2% 的框在保留图里找不到任何 IoU ≥ 0.1 的目标。

#### 2.1.3 HazyDet

`{train,val,test}/{hazy_images,images}` 加 `*_coco.json`，另有 `real_world/{train,test}` 加 `*_real.json`。COCO 标注的是雾图，同名清晰图复用同一批框。

选图（源口径，未切块）：

| 类别 | 图数 | 框数 | 处理 |
| --- | --- | --- | --- |
| 清晰图（与雾图同名同框） | 5,907 | 234,354 | 收，雾版丢弃 |
| 合成雾，无同名清晰图 | 5,093 | 130,079 | 收，只有这一份 |
| 真实雾 | 600 | 19,296 | 收（与 RDDTS 同一批，MD5 全同） |
| 合成雾，有同名清晰图 | 5,907 | 234,354 | 不收，同一批实例的第二个域 |

合计源图 11,543 张 / 380,128 框（候选 11,600，剔除同帧双标注 55 张、清洗后变空 2 张）。**不去重**：并集比值中位 1.48，可安全删的仅 0.4%。

切块后（入池口径）：

| 项目 | 统计 |
| --- | --- |
| 图 / 框 | **27,399 / 445,247** |
| 逐类 | car 408,950、truck 16,733、bus 19,564 |
| 占盘 | 6.3 GB |

边长分布（源框口径）：<8px 2.3%、8–16 11.3%、16–32 23.9%、32–64 37.4%、≥64 25.1%。

#### 2.1.4 CODrone

10,004 张全部 3840×2160，`annfile` 是 DOTA 风格 txt（八点 OBB + 类别），`labels/` 是同内容 XML，不用。

源图 10,004 张 / 443,583 框、12 类，边长 <8px 0.7% / 8–16 8.2% / 16–32 26.9% / 32–64 33.9% / ≥64 30.5%。

切块后（入池口径）：

| 项目 | 统计 |
| --- | --- |
| 图 / 框 | **43,440 / 532,613** |
| 逐类 | car 272,361、people 96,402、motor 89,261、truck 29,108、traffic sign 13,733、traffic light 8,184、boat 8,627、bus 5,750、bicycle 4,707、tricycle 3,480、ship 522、bridge 478 |
| 占盘 | 14 GB |

场景：50 个，昼 6,097 / 夜 3,714，高度 30/60/100 m，视角 30°/90° 各半。

#### 2.1.5 ATR-UMOD

`{train,test}/{images,labels}`，640×512，红外半边已删，本地只剩 RGB。每图一个 XML，每个 object 带 `robndbox`（cx/cy/w/h/angle）与 `polygon`（四点），取多边形的 min/max 转水平框。

| 项目 | 统计（实测） |
| --- | --- |
| 图 / 框 | **13,245 / 161,787**（13,353 图中 108 个空标注跳过、12 个退化框丢弃） |
| 逐类 | car 69,340、SUV 36,481、van 18,151、bus 9,797、freight car 9,746、truck 8,553、motor 3,631、tank truck 1,879、trailer 1,870、excavator 1,287、crane 1,052 |
| 目标边长 | <8px 4.8%、8–16 28.9%、16–32 48.9%、32–64 15.2%、≥64 2.1% |
| 分辨率 | 640×512，≤2560 不切块 |

**不去重**：相邻编号帧的并集比中位 1.71、A→B 框匹配率 0.00——重打包时文件名被重新编号，相邻编号不是同一场景的相邻帧。

### 2.2 COD 训练

#### 2.2.0 图源与管线

COD 图源只有一批：**SkyFind 剔四域 + 同帧去重后的 15,432 张**，清单 `SkyFind/cod.json`（`{root, images, stats, 阈值}`），下游只读它。FOD 五集 1.97M 实例不动。按源统计、划分实测、下载解压与去重细节见 [2026-11-08-uav-skyfind.md](2026-11-08-uav-skyfind.md)。

| | 图数 | 说明 |
| --- | ---: | --- |
| SkyFind 全量 | 35,599 | 解压到 `SkyFind/train/` + `SkyFind/test/` |
| − VisDrone | 4,688 | FOD 重复 + 可能含 test-dev 帧，考卷一泄漏险 |
| − DroneVehicle | 4,829 | FOD 重复 |
| − SeaDronesSee / MOBDrone | 4,736 | 考卷二整集留出，已移到 `SkyFind/test/` |
| 剩（21,346）| 21,346 | 自爬 8,216 + UVSD 3,950 + AU-AIR 2,280 + UAVDT 1,934 + Okutama 1,521 + CARPK 1,221 + Stanford 1,170 + 分割四域 1,054 |
| **− 同帧去重** | −5,914 | dHash 汉明 ≤6，按源内部、簇内取最清晰那张（`data_prep/skyfind_dedup.py`） |
| **进 COD** | **15,432** | 清单 `SkyFind/cod.json` |

标注全由我们自己的管线产出：Uni 全召回（wedetect_uni，16px 以下切片推理）→ VLM `coarse/mid/fine` 三段命名 → 置信度分层（高分直收、低分人审、16px 以下全人工）。进管线前对 FOD 池跑 pHash 兜底。

#### 2.2.1 产出（待跑）

- 预估框量待 Uni 实测；按穷尽标注口径约 30–60 万框（15,432 图 × 20–40 目标），不是 SkyFind 自报的每图 ~10
- 产出标注属 SkyFind 衍生物（cc-by-nc-nd-4.0），只自用训练，不进发布包
