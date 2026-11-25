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

## 2. 训练集准备

### 2.1 FOD 训练集

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

| 数据集 | 入池 json | 类数 | 图 / 框 | 转换脚本 |
| --- | --- | --- | --- | --- |
| VisDrone | `visdrone_trainval.json` | 10 | 7,019 / 381,963 | `visdrone2coco.py` |
| DroneVehicle | `dronevehicle.json` | 5 | 27,654 / 452,568 | `dronevehicle2coco.py` |
| HazyDet | `hazydet.json` | 3 | 11,543 / 380,128 | `hazydet2coco.py` |
| CODrone | `codrone.json` | 12 | 10,004 / 443,583 | `codrone2coco.py` |
| ATR-UMOD | `atrumod.json` | 11 | 13,245 / 161,787 | `atrumod2coco.py` |
| **合计** | 5 份 | 21 | **69,465 / 1,820,029** | — |

**1,820,029 是训练实例数**（五集 annotations 总数，dataloader 真正看到的条数），对外发布与命名都用它。全图入池，切块只在方法侧训练与推理时按需做，tile 不落盘。类数相加 41、去重后 21。

命名：`{dataset}.json` 是入池标注，`{dataset}_class_texts.json` 是类名表。

**Uni objectness 微调按边切 1280**：任一边 >1600 的图切块（切边恒 1280 规则在 `slice_coco.py`，`python3 data_prep/slice_coco.py --ann <ds>.json --dataset <DATASET>`）；mosaic 会把 >1280 子图缩到 1280，切块避免丢小目标。

| 数据集 | 微调用 json | 图 / 框 | 处置 |
| --- | --- | --- | --- |
| VisDrone | `visdrone_trainval_tiles.json` | 8,679 / 436,904 | 1,667 张 >1600 切块（2000×1500 / 1920×1080 / 1916×1078）|
| DroneVehicle | `dronevehicle.json` | 27,654 / 452,568 | 全 840×712，不切 |
| HazyDet | `hazydet_tiles.json` | 30,169 / 480,690 | 2,866 张 1601–2560 带切块，其余原图登记 |
| CODrone | `codrone_tiles.json` | 43,440 / 532,613 | 全 3840×2160 |
| ATR-UMOD | `atrumod.json` | 13,245 / 161,787 | 全 640×512，不切 |
| **合计** | 5 份 | **123,187 / 2,064,562** | — |

**格式与路径**：只吃标准 COCO（`WeCocoDataset`→`YOLOv5CocoDataset`→`mmdet.CocoDataset`），框一律 `[x, y, w, h]`；`file_name` 相对 `/mnt/sda1/datasets/drone` 且带集名前缀，全池共用一个 `data_root`。

**类名规则**（词表在这张表里定，各数据集小节不再逐个评论）：只用空格分隔，禁连字符与下划线；同一概念在不同源必须写成同一个词，否则合并词表时裂成两个类、两个 embedding；错拼由各自 ALIAS 收敛。

| 源写法 | 池内写法 | 出处 |
| --- | --- | --- |
| `traffic-sign`、`traffic-light` | `traffic sign`、`traffic light` | CODrone |
| `awning-tricycle` | `awning tricycle` | VisDrone |
| `freight_car` | `freight car` | DroneVehicle、ATR-UMOD |
| `tank_truck` | `tank truck` | ATR-UMOD |
| `motor` | `motorcycle` | CODrone 原始名；ATR 原始名本就是 motorcycle、不改。标准名见 §2.3 |
| `suv` | `SUV` | ATR-UMOD，缩写按论文写法；两种写法 embedding 余弦 0.980，无精度差 |

合并后 21 词，无连字符与下划线残留。源内划分 `pedestrian` / `people`（VisDrone）、`boat` / `ship`（CODrone，7,143 / 434 框）在 §2.3 统一词表里都保留为独立类。写法差异的 embedding 距离用 `data_prep/check_class_tokenizer.py` 实测。

| 原始形态 | 数据集 | 转换 |
| --- | --- | --- |
| 水平框 | VisDrone、DroneVehicle、HazyDet | 坐标直映射 |
| 旋转框 | CODrone、ATR-UMOD | 四顶点 min/max 取外接矩形 |
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
| 逐类框数 | car 158,930、pedestrian 88,181、motorcycle 34,533、people 32,184、van 26,931、truck 13,625、bicycle 11,767、bus 6,177、tricycle 5,857、awning tricycle 3,778 |
| 截断 / 遮挡 | 截断 14,845（3.9%）；遮挡 无 184,113 / 部分 161,112 / 严重 36,738（9.6%） |
| 目标边长 | <8px 39,511（10.3%）、8–16 103,475（27.1%）、16–32 128,931（33.8%）、32–64 77,352（20.3%）、≥64 32,694（8.6%） |
| 分辨率 | 长边最大 2000；前六档 1400×1050（2,498）、1400×788（1,299）、1360×765（1,151）、2000×1500（772）、1916×1078（537）、960×540（371） |

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

选图：

| 类别 | 图数（候选） | 框数（候选） | 处理 |
| --- | --- | --- | --- |
| 清晰图（与雾图同名同框） | 5,907 | 234,354 | 收，雾版丢弃 |
| 合成雾，无同名清晰图 | 5,093 | 130,079 | 收，只有这一份 |
| 真实雾 | 600 | 19,296 | 收（与 RDDTS 同一批，MD5 全同） |
| 合成雾，有同名清晰图 | 5,907 | 234,354 | 不收，同一批实例的第二个域 |

合计 11,543 张 / 380,128 框（候选 11,600，剔除同帧双标注 55 张、清洗后变空 2 张）。**不去重**：并集比值中位 1.48，可安全删的仅 0.4%。

边长分布：<8px 2.3%、8–16 11.3%、16–32 23.9%、32–64 37.4%、≥64 25.1%。

#### 2.1.4 CODrone

10,004 张全部 3840×2160，`annfile` 是 DOTA 风格 txt（八点 OBB + 类别），`labels/` 是同内容 XML，不用。

10,004 张 / 443,583 框、12 类，边长 <8px 0.7% / 8–16 8.2% / 16–32 26.9% / 32–64 33.9% / ≥64 30.5%。

场景：46 个（文件名前缀去重），昼 6,121 / 夜 3,883，高度 30/60/100 m，视角 30°/90° 各半（4,960 / 5,044）。

#### 2.1.5 ATR-UMOD

`{train,test}/{images,labels}`，640×512，红外半边已删，本地只剩 RGB。每图一个 XML，每个 object 带 `robndbox`（cx/cy/w/h/angle）与 `polygon`（四点），取多边形的 min/max 转水平框。

| 项目 | 统计（实测） |
| --- | --- |
| 图 / 框 | **13,245 / 161,787**（13,353 图中 108 个空标注跳过、12 个退化框丢弃） |
| 逐类 | car 69,340、SUV 36,481、van 18,151、bus 9,797、freight car 9,746、truck 8,553、motorcycle 3,631、tank truck 1,879、trailer 1,870、excavator 1,287、crane 1,052 |
| 目标边长 | <8px 4.8%、8–16 28.9%、16–32 48.9%、32–64 15.2%、≥64 2.1% |
| 分辨率 | 640×512 |

**不去重**：相邻编号帧的并集比中位 1.71、A→B 框匹配率 0.00——重打包时文件名被重新编号，相邻编号不是同一场景的相邻帧。

### 2.2 COD 训练集

#### 2.2.0 图源与管线

COD 图源只有一批：**SkyFind 剔四域 + 同帧去重 + VLM 航拍过滤后的 14,716 张**，清单 `SkyFind/cod_filtered.json`（`{root, images, stats, 阈值}`），下游只读它。FOD 五集 1.82M 框不动。按源统计、划分实测、下载解压与去重细节见 [2026-11-08-uav-skyfind.md](2026-11-08-uav-skyfind.md)；自爬采集脚本 `pexels.py` / `bilibili.py` / `video_frames.py`。

| | 图数 | 说明 |
| --- | ---: | --- |
| SkyFind 全量 | 35,599 | 解压到 `SkyFind/train/` + `SkyFind/test/` |
| − VisDrone | 4,688 | FOD 重复 + 可能含 test-dev 帧，考卷一泄漏险 |
| − DroneVehicle | 4,829 | FOD 重复 |
| − SeaDronesSee / MOBDrone | 4,736 | 考卷二整集留出，已移到 `SkyFind/test/` |
| 剩 | 21,346 | 自爬 8,216 + UVSD 3,950 + AU-AIR 2,280 + UAVDT 1,934 + Okutama 1,521 + CARPK 1,221 + Stanford 1,170 + 分割四域 1,054 |
| − 同帧去重 | −5,914 | dHash 汉明 ≤6，按源内部、簇内取最清晰那张（`data_prep/skyfind_dedup.py`） |
| − VLM 航拍过滤 | −716 | 自爬地拍 714 + 坏图 2（`data_prep/crawl_vlm_filter.py`，DeepSeek-V4.1-Flash 判视角，20 张人工锚点对拍全一致，16 张抽查 0 污染；逐图标注 `SkyFind/crawl_labels.json`） |
| **进 COD** | **14,716** | 清单 `SkyFind/cod_filtered.json`，召回结果 `cod_uni_boxes_fodcoco.jsonl` 已同步剔除对应行；最终有框入 `cod.json` 14,556 张（160 张零框剔除，见 §2.3） |

#### 2.2.1 产出流程

仿 WeDetect 数据引擎，标注器用裸 Qwen3-VL prompting。四个生产步骤 + 一次合并，口径与实测在 2.2.1.1–2.2.1.5；VLM 命名的拒识机制即唯一质检关，过检的框不再分层复核：

1. Uni 全召回（→ 2.2.1.1）
2. SAM 3 框提质（→ 2.2.1.2）
3. VLM 按框坐标命名（→ 2.2.1.3）
4. 概念补召回（→ 2.2.1.4）
5. 框全集合并（→ 2.2.1.5）


##### 2.2.1.1 Uni 召回实测与微调决策

Large-Uni 全量出 objectness 框。长边 >1280 内存切 1280×1280（模型输入 1280，tile 原生尺度推理），框映射回原图坐标；后处理：每 tile NMS 0.7 → top-500 截断 → 跨 tile 并集 → score>0.1 → NMS 0.5。输出 `SkyFind/cod_uni_boxes_fodcoco.jsonl`，每图一行 `{file, width, height, boxes: [[x,y,w,h,score],…]}`。断点续跑，双卡分片：

```bash
cd ~/projects/research/WeDetect
CUDA_VISIBLE_DEVICES=2 SHARD=0 .venv/bin/python data_prep/cod_uni_recall.py
CUDA_VISIBLE_DEVICES=3 SHARD=1 .venv/bin/python data_prep/cod_uni_recall.py
```

**第一步：纯 FOD 微调**

- 微调前（`cod_uni_boxes.jsonl`）：153 万框 17.2% 大框（>512px），分数与正常框重合（0.23 vs 0.25）阈值杀不掉——Uni 没见过航拍，实例粒度错乱
- 修法：FOD 69,465 图 / 182 万框（类别并 0 当 objectness）解冻 head/neck 微调 Uni，配方 `wedetect_uni_large_finetune.py`（large + 1280）
- objectness json：`data_prep/fod_objectness.py` 拼 5 个微调 json → `coco/fod_objectness.json`（123,187 图 / 2,064,562 框 / category 全 0）
- ckpt `wedetect_large_uni_fod.pth`，全量复召 `cod_uni_boxes_fod.jsonl`：尺度先验被拉向小目标，UVSD 70% 图含碎片大框，FOD 缺 >256px 尺度段
- 抽 50 张快速对比：`data_prep/fod_recheck.py` → `recheck_base.jsonl` / `recheck_fod.jsonl`

**第二步：FOD + COCO replay 重训**

- 修法：COCO replay，`make_coco_replay.py` 严格口径（COCO 图全部标注框 ∈ 22 类白名单：交通工具/交通设施/动物，与 FOD 词表重合），31,623 图 15.9 万框，`ConcatDataset` 挂 FOD+COCO 共 154,810 图，4 epochs
- ckpt `wedetect_large_uni_fodcoco.pth`，全量复召 `cod_uni_boxes_fodcoco.jsonl`

| 全量召回（14,709 有效图，均 NMS 0.5） | 框/图 | 大框/图 | 大框占比 | 分数中位 | 大框分中位 | UVSD 大框/图 |
|---|---:|---:|---:|---:|---:|---:|
| base | 90.4 | 13.92 | 15.4% | 0.259 | 0.250 | 5.25 |
| fod | 38.9 | 2.97 | 7.6% | 0.505 | 0.780 | 2.82 |
| fodcoco | 39.1 | 3.16 | 8.1% | 0.504 | 0.898 | 2.91 |
- 结论：大框量基本没回来（2.97→3.16，base 是 5.25），收益全在置信度（大框分中位 0.78→0.90）；fodcoco 更少更自信，下游 SAM 3 + Qwen 用它
- fodcoco 优势：框的质量更高，密集场景剪低分垃圾框（Uavid 117→69）、稀疏场景找回漏检（Semantic 22→29）

训练命令（tmux 里跑，续训加 `--resume`）：

```bash
CUDA_VISIBLE_DEVICES=2,3 .venv/bin/torchrun --nproc_per_node=2 train.py config/wedetect_uni_large_finetune.py --launcher pytorch
```

##### 2.2.1.2 SAM 3 框提质

权重走 `facebook/sam3`，脚本 `data_prep/sam3_recheck.py`；输入 `cod_uni_boxes_fodcoco.jsonl`（575,568 框），输出 `cod_uni_boxes_fodcoco_sam3.jsonl` 同格式、score 沿用。全部召回框无尺寸过滤：

- 整图 + 全框一次前向 box-prompt 出 mask，不切图（tile 边界截断框对紧框是硬伤；代价是 4K 图 <64px 小框的 mask 被 1008 压缩打残，靠步骤 3 拒识兜底）
- 短边 <8px 原样直通（288 分辨率下 mask 不足 1 像素，认领/收缩/碎框判定全是噪声）
- 其余框按 IoU≥0.3 认领最佳实例，`mask2bbox` 只取框内前景反算紧框替换原框
- 框内空 mask、零延展（线状）mask 丢弃；无响应框与认领 mask 有重叠即碎框丢弃，完全孤立保留原框；垃圾碎片由 merge 的 <4px 过滤兜底

```bash
modelscope download --model facebook/sam3 --local_dir checkpoints/sam3
# 独立环境：transformers 5（唯一带 Sam3 的版本，4.x 无）要求 torch ≥2.6，
# 主 venv 的 torch 2.5.1 被 mmcv 编译 ABI 钉死不能升，只能单独带
uv venv .venv-sam3 --python 3.12 && source .venv-sam3/bin/activate
uv pip install "torch>=2.6" torchvision -i https://mirrors.aliyun.com/pypi/simple/
uv pip install "transformers>=5"
```

全量运行（双卡分片）：

```bash
CUDA_VISIBLE_DEVICES=2 SHARD=0 NUM_SHARDS=2 .venv-sam3/bin/python data_prep/sam3_recheck.py
CUDA_VISIBLE_DEVICES=3 SHARD=1 NUM_SHARDS=2 .venv-sam3/bin/python data_prep/sam3_recheck.py
```

全量实测（2026-11-15）：

| | 数值 |
| --- | ---: |
| 行数 | 14,716（含 7 缺图透传） |
| 输入框 | 575,568 |
| 输出框 | 534,252（保留 92.8%） |
| 丢弃 | 41,316（无响应碎框 + 空mask） |
| 退化框 | 7 个 ≤1px 框，留给步骤 3 拒识；实际拒识 7、漏过 5（3 个零面积由 merge_boxes 剔除，2 个 1px 框留存） |


##### 2.2.1.3 VLM 按框坐标命名

输入 `SkyFind/cod_uni_boxes_fodcoco_sam3.jsonl`（534,252 框），一框一调用（WeDetect 数据引擎口径，`data_prep/vlm_name_boxes.py`）：

- 模型：本地 vLLM `Qwen3.8-27B-FP8`；坐标写 `Box: (x1, y1, x2, y2)`，单位是发送图像的绝对像素（Qwen2.5-VL 起官方约定）
- 送图：裁剪窗 3 倍框（最小 128px，窗口随框走），小于 448px 上采样到 448（小目标多摊 token）；目标用品红外扩 3px 描边（零像素覆盖、品红非自然色不污染判断；WeDetect 同款锚点思路，其靠 SFT 适应高亮）；定位 = 文本坐标 + 画框双信号
- 三段口径：coarse=大类（vehicle/person/boat…）、mid=常见类型（car/van/truck/motor…）、fine=细粒度变体（sedan/hatchback/SUV…）；fine 不确定回填 mid，下游取 fine（缺则 mid）当类名
- 拒识：框内没有以它为主体的单一物体（空框/一片区域/一群物体）→ 剔除；贴边截断但可辨认 → 正常标注
- 并发：两段流水线——预处理池（4 线程，读图/裁剪/编码）+ 发送池（14 线程，纯网络 IO，Session 连接复用），逐框队列缓冲，`as_completed` 平滑供给
- 输出 `SkyFind/cod_uni_boxes_fodcoco_sam3_named.jsonl`：一行一框 `{file, i, box, score, coarse, mid, fine}`，拒识行 `{file, i, box, score, reject: true}` 也落盘（续跑不重问，下游过滤 reject 行）；断点 = 已含的 `(file, i)`，失败框不落盘续跑自动补

全量实测（2026-11-19）：

| | 数值 |
| --- | ---: |
| 输入框 | 534,252 |
| 成功命名 | 500,203（93.6%） |
| 拒识丢弃 | 34,049（6.4%） |
| 守恒 | 输出 = 输入，0 缺失 |


| 词表 | 原始词数 | 清洗后 | top |
| --- | ---: | ---: | --- |
| coarse | 62 | —（训练不用） | vehicle 407,960 / person 60,155 / boat 25,775 / animal 3,078 |
| mid | 316 | 230（剔 86 个 JUNK） | car 342,609 / pedestrian 56,271 / boat 27,795 / truck 19,984 |
| fine | 437 | **288（管线类名，训练词表 174 见 §2.3）** | sedan 175,324 / car 130,394 / pedestrian 56,257 / suv 21,384 |

清洗（`data_prep/normalize_vocab.py`，输入 `named.jsonl`）：同义词合并 + 拼写修复 + motor 按 coarse 分流（vehicle→motorcycle / boat→motorboat）+ JUNK 剔除。2026-11-23 建筑家族九词（house/building/apartment/shed/warehouse/cabin/hut/church/tower/factory）入 JUNK 整体剔除：俯拍下类间边界无判据、标注覆盖任意、SAM3 recover 对 house 粒度漂移，当背景 stuff。训练类名取 fine（脏则回退 mid）。

输出 `SkyFind/cod_uni_boxes_fodcoco_sam3_named_norm.jsonl`：499,506 框 / 288 类 / 零 reject 行（训练词表经 §2.3 尾类归并后 174 类）。头部陡：sedan+car+pedestrian 占 72.5%，尾类见 top40。

##### 2.2.1.4 SAM3 概念补漏

输入 `named_norm.jsonl`（499,506 框），`data_prep/sam3_recover.py` 4 卡并行：每图喂该图已有框出现过的 fine 概念词，SAM3 1008 tile 扫图；SCORE 0.5 → 跨 tile/概念 NMS 0.5 → 贴切线截断框剔除 → 与已有框 IoU 0.3 去重。只补图中已有类型的漏检实例。

输出 `SkyFind/cod_uni_boxes_fodcoco_sam3_named_norm_recover.jsonl`（`{file, i, box, score, label}`，i 续编），与 named_norm 合计 **630,574 框 / 288 类**（<4px 框处置见 2.2.1.5）。2026-11-23 建筑九词入 JUNK 后同步过滤 recover（剔 11,489 行）。

| | 数值 |
| --- | ---: |
| 输入图 | 14,557 |
| 补出新框 | 131,323 |
| 覆盖图 | 9,713（66.7%） |
| 每图新增 | 中位 5 / p90 34 / max 404 |
| score 中位 | 0.635 |
| top | car 48,575 / pedestrian 29,966 / boat 10,379 / sign 7,334 |

##### 2.2.1.5 框全集合并

输入 `named_norm.jsonl`（499,506）+ `recover.jsonl`（131,323），`data_prep/merge_boxes.py`：

- 统一行 `{file, i, box, score, label}`，label = 训练类名 fine
- <4px 框 792 个：537 个能回溯到 ≥4px Uni 原框的救援回原框几何（recheck 紧框收缩误伤，label 仍是 VLM 验证过的）；Uni 原框也 <4px 的 149 个与 recover 自检的 106 个剔除
- 输出 `..._all.jsonl` + `..._all_classes.jsonl`（288 类按名排序固定 id，仅 COD 内部排序用；训练共用 §2.3 的 `vocab_merged.json`）

转 COCO 只认这两个文件。

| | 数值 |
| --- | ---: |
| named_norm | 499,506 |
| recover | 131,323 |
| <4px 框救援 | 537 |
| <4px 框剔除 | 255 |
| 全集 | 630,574 框 / 288 类 |
| top | car 178,848 / sedan 175,721 / pedestrian 86,115 / boat 27,908 |

### 2.3 COD + FOD 训练集

统一词表 **174 类**（COD 288 + FOD 独有 4 类，其中 118 个 ≤4 框尾类归并或剔除），落盘 `data/coco_anno/vocab_merged.json`，归并表 `data/coco_anno/vocab_tail_map.json`。6 个单集 json 的 category_id 全部用统一 id，同名即同类。

| 处理 | 内容 |
| --- | --- |
| 同名直接合并（17 类） | car / truck / bus / van / bicycle / tricycle / boat / ship / pedestrian / motorcycle / SUV / crane / excavator / trailer / traffic sign / traffic light / tank truck |
| 尾类归并（118 类） | 每个类在 ovd 里只有 1–4 框：75 类并入上位类（133 框，如 giraffe→animal、gondola→boat、military ship→ship），43 类丢弃（87 框，家居物件/误标/飞行器细分） |
| COD 改名 2 处 | suv→SUV（21,666 框；另有 6 框尾类并入，SUV 合计 21,672）；tanker truck→tank truck（25 框） |
| FOD 改名 1 处 | motor→motorcycle：在 `visdrone2coco.py` / `codrone2coco.py` / `atrumod2coco.py` 源头改掉（VisDrone 原生类别索引、CODrone 原始名 motor、ATR 原始名 motorcycle），bbox 坐标不动 |
| 保持分离 | people（人群）≠ pedestrian（单人）；FOD freight car（货车）≠ COD train（铁道）；sign 与 traffic sign 分开 |

1. `data_prep/build_unified_vocab.py`：读 SkyFind 288 类表 + FOD 5 json 的 categories + `vocab_tail_map.json`（人工审定的归并/剔除表，重跑只读），产出 `data/coco_anno/vocab_merged.json`（174 类）
2. `data_prep/{visdrone,codrone,dronevehicle,hazydet,atrumod}2coco.py`：各自读 `vocab_merged.json`，categories 与框的 category_id 直接写统一 id（FOD 21 类无尾类，不经归并表），产出 5 个 json + `*_class_texts.json`（数量口径见 2.1）
3. `data_prep/cod2coco.py`：输入 `vocab_merged.json` + `vocab_tail_map.json` + `SkyFind/cod_uni_boxes_fodcoco_sam3_named_norm_all.jsonl`（label 先查归并表再取 id）+ `cod_uni_boxes_fodcoco.jsonl`（宽高字典）；输出 `data/coco_anno/cod.json` + `cod_class_texts.json`（14,556 图 / 630,487 框 / 170 类，file_name `SkyFind/train/<file>`，160 张零框剔除）
4. `data_prep/merge_coco.py`：拼训练入口，只留标准字段（image: id/file_name/width/height；ann: id/image_id/category_id/bbox/area/iscrowd），image_id/ann_id 重编；输出 `fod.json` + `fod_class_texts.json`（5 集 69,465 图 / 1,820,029 框 / 21 类）、`ovd.json` + `ovd_class_texts.json`（cod + fod 84,021 图 / 2,450,516 框 / 174 类）

按此顺序重跑，产物 md5 与当前一致（2026-11-24 实测）。

训练按实验指定 `cod.json` / `fod.json` / `ovd.json` 三档入口；单集 json 保留附加字段（score/truncate/occlusion/ignore_regions）。

规模对比：

| | 图 | 框 | 框/图 | 类 |
| --- | ---: | ---: | ---: | ---: |
| fod | 69,465 | 1,820,029 | 26.2 | 21 |
| cod | 14,556 | 630,487 | 43.3 | 170 |
| ovd | 84,021 | 2,450,516 | 29.2 | 174 |
| COCO train2017 | 118,287 | 860,001 | 7.3 | 80 |
| LVIS v1 train | 100,170 | ~1.27M | 12.7 | 1,203 |

- LVIS 图出自 COCO 2017 全集 164k，重切为 train 100,170 / val 19,809 / test 19,822
- 图数 COCO 的 71%、框数 2.85 倍、密度 4 倍；类数 COCO 的 2.2 倍、LVIS 的 1/7
- 短板在 COD 图量：14,556 张（COCO 的 1/8），仅一批 SkyFind 源
- COD 头部陡：car+sedan+pedestrian 占 69.9%
- ovd 目标边长 <32px 占 52.2%：

| 边长 | <8px | 8–16 | 16–32 | 32–64 | ≥64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 占比 | 3.7% | 15.1% | 33.4% | 30.4% | 17.3% |

## 3. 评测集准备

### 3.1 VisDrone test-dev

1,610 图 / 75,102 框 / 10 类，`data/coco_anno/visdrone_testdev.json`（统一词表 id；类 0 进 `ignore_regions`、类 11 进 `others_regions`），脚本 `data_prep/visdrone2coco.py --split testdev`。

| 项目 | 数值 |
| --- | ---: |
| 图 / 框 | 1,610 / 75,102 |
| 逐类 | car 28,074、pedestrian 21,006、people 6,376、motorcycle 5,845、van 5,771、bus 2,940、truck 2,659、bicycle 1,302、awning tricycle 599、tricycle 530 |

口径：GT 不穷尽，检出未标注的真实目标计误报（零样本摸底 AP 9.6 偏低的原因之一）。

### 3.2 SeaDronesSee + MOBDrone（海洋卷）



## 4. 海洋二集准备

结论：MOBDrone / SeaDronesSee 的 train 全量转 COCO 入 FOD；标签映射进统一词表，恢复 `lifebuoy`。

### 4.1 MOBDrone

原始：66 段视频，1920×1012、29.97fps，逐帧抽出 126,170 图，帧号 = 视频解码序号。

标注：`annotations_5_custom_classes.json`（5 类 person/boat/surfboard/wood/life_buoy，181,689 框）用这份；`annotations_3_coco_classes.json` 不用。

```bash
mkdir -p /mnt/sda1/datasets/drone/MOBDrone && cd /mnt/sda1/datasets/drone/MOBDrone
wget -c "https://zenodo.org/records/5996890/files/annotations_5_custom_classes.json?download=1" -O annotations_5_custom_classes.json
wget -c "https://zenodo.org/records/5996890/files/MOBDrone_videos.zip?download=1" -O MOBDrone_videos.zip
```

#### 4.1.1 抽帧口径

两处特例：`DJI_0915_0007_40.mp4` 对应标注流 `DJI_0915_0007_40m`；`DJI_0804_0017_40m` 视频 1,593 帧 > 标注 1,589 帧，尾部 4 帧不评测。

官方逐帧全抽、冗余极高（相邻帧只有 4–6% 新框），按 `idx % 10 == 0` 抽帧，无框帧不入。

| k=10 | 图 | 框 |
| --- | ---: | ---: |
| test | 2,398 | 8,983 |
| train | 6,730 | 9,221 |
| 合计 | 9,128 | 18,204 |

逐类 test person 4,630 / boat 2,151 / life_buoy 1,044 / wood 963 / surfboard 195，train person 6,720 / boat 1,860 / wood 641。词表 174 → 176：恢复 `lifebuoy`，新增 `driftwood`（原 `wood` 海面漂浮木）。

高度分布（官方全帧）：

| 高度 | train 帧 | train 框 | test 帧 | test 框 |
| --- | ---: | ---: | ---: | ---: |
| 10m | — | — | 958 | 151 |
| 20m | 5,114 | 3,157 | 4,939 | 6,116 |
| 30m | 16,223 | 13,117 | 13,181 | 18,995 |
| 40m | 24,857 | 24,843 | 8,189 | 16,676 |
| 50m | 20,514 | 24,812 | 8,669 | 42,572 |
| 60m | 21,858 | 26,148 | 1,668 | 5,102 |
| **总计** | **88,566** | **92,077** | **37,604** | **89,612** |

#### 4.1.2 抽帧、打包与传回

本机两份清单 `keep_frames.txt`（真实帧号）与 `keep_seq.txt`（抽帧顺序号）各 9,128 行，打包只传后者。服务器抽帧文件名 `frames/<流名>_%06d.jpg` 为顺序号；本机解压后按 `真实帧号 = 顺序号 × 10` 重命名为 `<流名>_%06d.PNG`（特例流 `40` → `40m`），与标注 `file_name` 完全一致：9,128 张全部命中、零空标注。

服务器抽帧：

```bash
cd ~/video_split_fullhd && mkdir -p frames
i=0; tot=$(ls DJI_*.mp4 | wc -l)
for v in DJI_*.mp4; do
  s=${v%.mp4}; i=$((i+1))
  [ -e "frames/${s}_000000.jpg" ] && continue
  ffmpeg -y -loglevel error -i "$v" -vf "select='eq(mod(n,10),0)'" -vsync 0 -start_number 0 -q:v 2 "frames/${s}_%06d.jpg"
done
ls frames | wc -l                 # 12,643
```

打包并经 GitHub 中转传回。本机传清单：

```bash
scp /mnt/sda1/datasets/drone/MOBDrone/keep_seq.txt g9918265@35.208.194.214:~/video_split_fullhd/
```

服务器打包（实测 1.9 GB）：

```bash
cd ~/video_split_fullhd && tar cf frames.tar -C frames -T keep_seq.txt
tar tf frames.tar | wc -l        # 9128
```

服务器 gh 交互式传 Release（先 `gh auth login`）：

```bash
REPO=hougeai/mobdrone-frames
cd ~/video_split_fullhd
gh release create mobdrone-frames frames.tar --repo $REPO --title "MOBDrone frames"
```

本机拉回：

```bash
cd /mnt/sda1/datasets/drone/MOBDrone
wget -c "https://ghfast.top/https://github.com/$REPO/releases/download/mobdrone-frames/frames.tar"
tar tf frames.tar | wc -l        # 9128
tar xf frames.tar -C frames
```

### 4.2 SeaDronesSee

图片：S-ODv2，14,227 图 = 8,930 train + 1,547 val + 3,750 test；是独立图像集不是视频抽帧
标注：6 类 swimmer/boat/jetski/life_saving_appliances/buoy/ignored；CC0 1.0）

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download --repo-type dataset dronefreak/SeaDronesSee --local-dir /mnt/sda1/datasets/drone/SeaDronesSee --max-workers 1
```

| split | 图片 | 框 |
|---|---:|---:|
| train | 8,930 | 57,760 |
| val | 1,547 | 9,630 |
| 合计 | 10,477 | 67,390 |

标注在 `data/images/{train,val}/shard_*/metadata.jsonl`，与 jpg 一一对应、无空标注。