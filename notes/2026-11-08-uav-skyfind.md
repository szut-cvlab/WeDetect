# SkyFind 复用

决策（2026-11-09 定稿）：**合并方案，不是全押 SkyFind**。FOD 五集 1.97M 实例为训练主体不动；SkyFind 全量 35,599 图**剔除四个域**后剩 **21,346 张**走 Uni 全召回 + VLM 层级命名 + 人审兜底，当 COD 补词表——**COD 图源全部来自 SkyFind**，数据笔记旧 §2.2.1「只取 8,189 自爬帧」口径作废。三张考卷沿用 plan 不变（VisDrone test-dev、SeaDronesSee+MOBDrone 留出、自建 3–5k）；自建考卷图源用 pexels 自爬（数据笔记 §2.3）。SkyFind 衍生物（切块/标注）受 cc-by-nc-nd-4.0 约束**不进发布包**，只随论文开源生成管线。

## 1. 全量图按源统计

统计来源：仓库根目录三个标注清单 —— `Train.json`（30,820 图）、`Test.json`（4,736 图）、`Val.json`（4,354 图），2026-11-08 从 `hf-mirror.com/datasets/kunyu241/SkyFind` 下载；按 `fileName` 首段字母归类、按图去重实测：

| 源 | 图数 | REC 记录 | 源性质 |
| --- | ---: | ---: | --- |
| 自爬（纯数字文件名） | 8,216 | 63,635 | 它自己爬的 |
| DroneVehicle | 4,829 | 38,018 | 检测集（FOD 在用） |
| VisDrone | 4,688 | 99,390 | 检测集（FOD 在用） |
| UVSD | 3,950 | 40,380 | 待查（缩写不常见） |
| SeaDronesSee | 3,671 | 14,570 | 海上无人机检测/分割 |
| AU-AIR | 2,280 | 15,698 | 检测集 |
| UAVDT | 1,934 | 30,079 | 检测/跟踪 |
| Okutama-Action | 1,521 | 9,190 | 航拍人群动作 |
| CARPK | 1,221 | 26,933 | 停车场车辆检测（UCSD 2017，无人机 40 m 航拍 4 个停车场） |
| Stanford Drone | 1,170 | 3,555 | 校园俯拍视频 |
| MOBDrone | 1,065 | 1,976 | 无人机移动目标 |
| Semantic-Drone | 385 | 2,058 | 语义分割 |
| Aeroscapes | 276 | 1,070 | 语义分割 |
| UAVID | 249 | 4,689 | 语义分割 |
| UDD | 144 | 1,669 | 语义分割 |
| **合计** | **35,599** | 352,910 | — |

README 口径：1,015,638 target-expression 对 / 352,910 目标 / 35,599 图，平均每图 ~10 个被标目标。

要点：
- **train_aug 与 train 完全同图**（30,820 张 100% 重合，仅 GPT-4 表达式增强三倍），图不新增，25 GB 的 train_aug parquet 不用下

## 2. 全量图获取

**已完成（2026-11-09）**：按官方 README 走 zip 路线——`bigfile_part_00~11.zip` 12 分卷（~59 GB，zip 只有图，REC 标注在根目录 JSON、只用 fileName 对齐）→ `cat` 合并 → 解压后改名 `train/`，实测 **35,599 张**与清单吻合。比 parquet 路线（train+test 共 68 GB）省。

### 要剔除四个域

| 前缀 | 图数 | 剔除原因 |
|---|---:|---|
| `Visdrone_*` | 4,688 | FOD 在用 |
| `DroneVehicle_*` | 4,829 | FOD 在用 |
| `SeaDronesSee*` | 3,671 | 考卷二整集留出 |
| `MOBDrone*` | 1,065 | 考卷二整集留出 |

```bash
cd /mnt/sda1/datasets/drone/SkyFind
mv images_raw train                    # train/ = COD 21,346 + Visdrone_*/DroneVehicle_* 存档帧
mkdir -p test
mv train/SeaDronesSee* train/MOBDrone* test/   # 考卷二图，留作 GT 标注用
ls train | grep -cv 'Visdrone_\|DroneVehicle_'   # 应剩 21,346
```

**硬规则**：所有下游（Uni 扫批、切块、dataloader）只取 `train/` 里**不带** `Visdrone_*` / `DroneVehicle_*` 前缀的图（21,346 张），过滤写死在脚本/清单里。

### 同帧去重（`data_prep/skyfind_dedup.py`）

21,346 张按**源内部**去重：dHash 汉明 ≤6 视为同帧（LSH 分带找候选，贪心分簇）→ 簇内按 Laplacian 方差取最清晰那张当代表 → 代表之间再按清晰度降序判一遍，保证保留集两两距离 >6。不跨源比对（校园俯拍在 Stanford 与自爬都有，跨源会误删）。

| 源 | 输入 | 保留 |
| --- | ---: | ---: |
| 自爬 | 8,216 | 6,833 |
| UVSD | 3,950 | 3,581 |
| AUAIR | 2,280 | 1,228 |
| UAVDT | 1,934 | 972 |
| Okutama-action | 1,521 | 1,046 |
| CAPRK | 1,221 | 684 |
| Stanford | 1,170 | 70 |
| Semantic | 385 | 382 |
| Aeroscapes | 276 | 254 |
| Uavid | 249 | 238 |
| UDD | 144 | 144 |
| **合计** | **21,346** | **15,432** |

产物只有 `SkyFind/cod.json`（`{root, images, stats, 阈值}`，**下游只读这份清单**）。

核对过的三点：保留集抽样最小距离 13（自爬）/ 7（Stanford），全部 >6；抽 5 组被删图与其代表，代表清晰度全部更高；Stanford 1,170 → 70 是校园固定航线同机位反复抽帧被合掉。
