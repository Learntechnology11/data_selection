# DiffusionDrive 在 nuScenes 上的指标复现指南

> 目标：复现 CVPR 2025 论文 **DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving** 在 nuScenes 上报告的开环规划指标。  
> 核验范围：论文、官方 `nusc` 分支、官方训练/评测配置、官方发布日志、SparseDrive 环境说明与 nuScenes devkit。  
> 核验日期：2026-07-20。

---

## 1. 先明确要复现的是什么

DiffusionDrive 在 nuScenes 上报告的是基于 SparseDrive 框架的 **开环 ego trajectory planning** 结果，核心指标为：

| 指标 | 1 s | 2 s | 3 s | Avg |
|---|---:|---:|---:|---:|
| L2（m） | 0.27 | 0.54 | 0.90 | 0.57 |
| Collision（%） | 0.03 | 0.05 | 0.16 | 0.08 |

官方训练日志中的未四舍五入结果为：

| evaluator 输出 | 0.5 s | 1.0 s | 1.5 s | 2.0 s | 2.5 s | 3.0 s | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `obj_box_col` | 0.039% | 0.029% | 0.039% | 0.054% | 0.082% | 0.156% | 0.080% |
| `L2` | 0.1688 | 0.2692 | 0.3918 | 0.5387 | 0.7096 | 0.9031 | 0.5704 |

### 1.1 这里的“测试集”实际上是 validation split

官方配置中：

```python
train.ann_file = "data/infos/nuscenes_infos_train.pkl"
val.ann_file   = "data/infos/nuscenes_infos_val.pkl"
test.ann_file  = "data/infos/nuscenes_infos_val.pkl"
```

因此，论文表格中的 nuScenes 规划结果是：

- 使用官方 `train` split 训练；
- 使用官方 `val` split 本地评测；
- **不是**使用无公开标签的 `v1.0-test` hidden test split；
- 配置中的 `test_mode=True` 只表示数据集处于推理模式，并不表示使用 nuScenes 官方 test split。

这是复现时最容易混淆的问题。

### 1.2 该指标不是 nuScenes 官方 detection leaderboard 指标

这里不评测 NDS、mAP，也不是 nuScenes 官方 prediction challenge 指标，而是沿用 ST-P3 / UniAD 系列工作中常见的开环规划评测：

- 未来 3 秒 ego 轨迹 L2；
- 预测 ego box 与未来目标框的碰撞率；
- 评测依赖未来 ego 真值轨迹和未来动态目标框，因此无法直接在无标签 test split 上本地计算。

---

## 2. 推荐采用“两阶段复现”

不要一开始就完整训练。建议按以下顺序执行。

### 阶段 A：官方权重指标验证

目的：

1. 验证环境、数据、info 文件、聚类锚点和 evaluator 均正确；
2. 使用作者发布的 checkpoint 重现约 `0.5704 m / 0.080%`；
3. 排除训练随机性。

这是必须先完成的基线。

### 阶段 B：完整 stage-2 重训练

目的：

1. 从 SparseDrive stage-1 checkpoint 初始化；
2. 按官方 8 GPU、global batch 48、10 epoch 配置训练；
3. 使用同一 evaluator 检查重训练结果是否接近论文。

只有阶段 A 成功后，阶段 B 的结果才具有可诊断性。

---

## 3. 代码版本必须固定

DiffusionDrive 默认 `main` 分支主要对应 NAVSIM。nuScenes 代码位于 `nusc` 分支。

### 3.1 最严格的版本固定方式

```bash
git clone https://github.com/hustvl/DiffusionDrive.git
cd DiffusionDrive

# 官方 nuScenes release tag，指向作者发布 nuScenes 代码时的提交
git checkout DiffusionDrive_nuScenes

git rev-parse HEAD
git status
```

也可以使用：

```bash
git checkout nusc
```

但此时应保存具体提交号：

```bash
git rev-parse HEAD | tee repro_git_commit.txt
```

### 3.2 不建议的做法

- 不要直接在当前 `main` 分支运行 nuScenes 配置；
- 不要混用 NAVSIM 环境和 `nusc` 分支环境；
- 不要在没有保存 commit SHA 的情况下修改源码后再比较论文指标；
- 不要先升级到最新 PyTorch、MMCV、MMDetection，再期待完全一致的结果。

---

## 4. 官方训练环境

作者发布的训练日志记录了以下实际环境：

| 项目 | 官方日志 |
|---|---|
| OS | Linux |
| Python | 3.8.19 |
| GPU | 8 × NVIDIA GeForce RTX 4090 |
| CUDA Toolkit / NVCC | 11.6 |
| PyTorch | 1.13.0+cu116 |
| TorchVision | 0.14.0+cu116 |
| MMCV | 1.7.1 |
| MMDetection | 2.28.2 |
| OpenCV | 4.8.1 |
| GCC | 9.3 |
| 训练峰值显存 | 日志约 15.3 GB / GPU |
| 单 iteration 时间 | 日志约 1.34 s |
| stage-2 训练时间 | 日志约 2 小时 13 分钟，不含环境与数据准备 |

上述时间只用于量级参考，受存储、CPU、数据缓存和 GPU 频率影响。

### 4.1 推荐建立独立 Conda 环境

```bash
conda create -n diffusiondrive_nusc python=3.8.19 -y
conda activate diffusiondrive_nusc

python -m pip install --upgrade pip setuptools wheel
```

安装官方日志中的 PyTorch：

```bash
pip install \
  torch==1.13.0+cu116 \
  torchvision==0.14.0+cu116 \
  torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

安装 MMCV：

```bash
pip install mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
```

安装仓库依赖：

```bash
pip install -r requirement.txt
pip install flash-attn==2.3.2 --no-build-isolation
```

### 4.2 `diffusers` 是官方依赖清单中的遗漏项

nuScenes 源码直接导入：

```python
from diffusers.schedulers import DDIMScheduler
```

但 `requirement.txt` 没有固定 `diffusers` 版本。因此：

```bash
pip install diffusers
```

随后必须保存实际解析到的版本：

```bash
python - <<'PY'
import diffusers
print("diffusers:", diffusers.__version__)
PY

pip freeze > environment_freeze.txt
```

注意：

- 作者没有公开固定的 `diffusers` 版本；
- 因此它是“完全重训练数值一致性”的一个残余不确定因素；
- 不要未经验证就把任意 `diffusers` 版本称为官方版本；
- 若新版本与 Python 3.8 不兼容，应选择 pip 能解析的 Python 3.8 兼容版本，并记录版本；
- 阶段 A 使用官方 checkpoint 成功，是判断该版本是否可用的最佳实践标准。

### 4.3 编译自定义 CUDA 算子

```bash
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../..
```

验证：

```bash
python - <<'PY'
import torch
import mmcv
import mmdet
import diffusers

from projects.mmdet3d_plugin.ops import deformable_aggregation_function

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("diffusers:", diffusers.__version__)
print("custom CUDA op: OK")
PY
```

### 4.4 平台注意事项

该项目依赖：

- NCCL 分布式训练；
- `mmcv-full` CUDA 扩展；
- 自定义 deformable aggregation CUDA 算子；
- FlashAttention。

因此推荐直接使用 Linux GPU 环境。原生 Windows 环境不适合作为严格复现环境；WSL2 也需要自行处理 CUDA、编译器和 ABI 兼容性。

---

## 5. nuScenes 数据准备

### 5.1 所需数据

完整复现至少需要：

1. nuScenes `v1.0-trainval`；
2. 所有 `samples`；
3. 所有 `sweeps`；
4. `maps`；
5. CAN bus expansion；
6. 由仓库 converter 生成的 train/val info PKL。

标准 nuScenes scene split 为：

- train：700 scenes；
- val：150 scenes；
- test：150 scenes。

本项目只使用 train 和 val 来计算论文规划指标。

### 5.2 推荐目录结构

```text
DiffusionDrive/
├── data/
│   ├── nuscenes/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── maps/
│   │   ├── can_bus/
│   │   ├── v1.0-trainval/
│   │   └── v1.0-mini/              # 可选，仅用于 smoke test
│   ├── infos/
│   │   ├── nuscenes_infos_train.pkl
│   │   └── nuscenes_infos_val.pkl
│   └── kmeans/
├── ckpts/
│   ├── resnet50-19c8e357.pth
│   ├── sparsedrive_stage1.pth
│   └── diffusiondrive_nusc_stage2.pth
└── projects/
```

创建软链接：

```bash
mkdir -p data
ln -s /absolute/path/to/nuscenes data/nuscenes
```

检查：

```bash
test -d data/nuscenes/samples
test -d data/nuscenes/sweeps
test -d data/nuscenes/maps
test -d data/nuscenes/can_bus
test -d data/nuscenes/v1.0-trainval
```

### 5.3 为什么训练时仍然需要 LiDAR

配置的推理模态为：

```python
use_camera=True
use_lidar=False
use_radar=False
```

但训练 pipeline 会读取 LiDAR 点云并生成多尺度深度监督：

```python
LoadPointsFromFile
MultiScaleDepthMapGenerator
```

因此：

- 模型推理是 camera-only；
- 完整训练仍需要 LiDAR 文件，用于辅助 depth supervision；
- 只下载相机图片无法完成官方训练流程；
- radar 不参与该配置。

### 5.4 生成 info 文件

```bash
bash scripts/create_data.sh
```

官方脚本会调用：

```bash
python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0
```

预期关键文件：

```text
data/infos/nuscenes_infos_train.pkl
data/infos/nuscenes_infos_val.pkl
```

### 5.5 严格检查样本数和关键字段

```bash
python - <<'PY'
import mmcv
import numpy as np

train = mmcv.load("data/infos/nuscenes_infos_train.pkl")
val = mmcv.load("data/infos/nuscenes_infos_val.pkl")

train_infos = train["infos"]
val_infos = val["infos"]

print("train samples:", len(train_infos))
print("val samples:", len(val_infos))
print("train metadata:", train.get("metadata"))
print("val metadata:", val.get("metadata"))

# 官方配置硬编码的训练样本数
assert len(train_infos) == 28130, (
    "train sample count is not 28130; do not train before checking converter/version"
)

required = [
    "cams",
    "lidar_path",
    "map_annos",
    "gt_ego_fut_trajs",
    "gt_ego_fut_masks",
    "gt_ego_fut_cmd",
    "fut_boxes",
]

for key in required:
    assert key in val_infos[0], f"missing field: {key}"

eligible = 0
for info in val_infos:
    mask = np.asarray(info["gt_ego_fut_masks"]).astype(bool)
    if mask.all():
        eligible += 1

print("planning-eval eligible val samples:", eligible)
print("data check: OK")
PY
```

注意：

- 官方配置把 train 长度硬编码为 `28130`；
- `num_iters_per_epoch` 也由这个数字计算，而不是运行时自动读取数据集长度；
- 如果你的 train PKL 不是 28130 条，不能直接继续训练；
- 应优先检查数据版本、converter 和生成过程，而不是只修改配置中的 `length`。

---

## 6. 聚类锚点：严格复现中的高风险点

模型需要以下聚类文件：

```text
data/kmeans/kmeans_det_900.npy
data/kmeans/kmeans_map_100.npy
data/kmeans/kmeans_motion_6.npy
data/kmeans/kmeans_plan_6.npy
```

### 6.1 严格复现优先使用作者 release 中的锚点

```bash
mkdir -p data/kmeans reference_logs

# 需要安装并登录可公开访问 GitHub 的 gh CLI；公开 release 通常无需账号授权
gh release download DiffusionDrive_nuScenes \
  --repo hustvl/DiffusionDrive \
  --pattern "*.npy" \
  --dir data/kmeans

gh release download DiffusionDrive_nuScenes \
  --repo hustvl/DiffusionDrive \
  --pattern "*.log*" \
  --dir reference_logs
```

也可以在 GitHub 的 `DiffusionDrive_nuScenes` release 页面手动下载全部 clustered anchors。

### 6.2 官方仓库存在一个需要注意的不一致

配置要求：

```text
data/kmeans/kmeans_plan_6.npy
```

但当前 `tools/kmeans/kmeans_plan.py` 中的示例脚本保存为：

```text
data/kmeans/kmeans_plan_vocab_6.npy
```

并且该脚本展示的是全局 `K=6` 聚类，而模型代码实际按驾驶命令选择规划 anchors。

因此，**不要把该脚本原样生成的文件重命名后就当作论文的 18 个规划 anchors**。

### 6.3 18 个 planning anchors 的含义

配置中：

```python
ego_fut_mode = 6
```

模型按 3 类导航命令选择每类 6 个 mode：

```text
3 commands × 6 modes = 18 planning anchors
```

严格复现时，`kmeans_plan_6.npy` 应与模型预期一致，典型逻辑形状为：

```text
(3, 6, 6, 2)
 └  └  └  └── x/y
 命令 mode 时间
```

验证：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

files = [
    "data/kmeans/kmeans_det_900.npy",
    "data/kmeans/kmeans_map_100.npy",
    "data/kmeans/kmeans_motion_6.npy",
    "data/kmeans/kmeans_plan_6.npy",
]

for p in files:
    assert Path(p).is_file(), f"missing: {p}"
    x = np.load(p)
    print(p, x.shape, x.dtype)

plan = np.load("data/kmeans/kmeans_plan_6.npy")
assert plan.shape == (3, 6, 6, 2), (
    f"unexpected plan anchor shape: {plan.shape}; "
    "prefer the author-released anchor file"
)
PY
```

如果 release 文件的实际形状与上述断言不同，应以对应 release commit 中模型源码的读取逻辑为准，不要强行 reshape。

### 6.4 若必须自行生成 anchors

至少满足：

- 只使用 `nuscenes_infos_train.pkl`；
- 不读取 val 轨迹，避免数据泄漏；
- planning anchor 按 3 类 command 分组；
- 每个 command 聚类 6 条 3 秒轨迹；
- 固定 KMeans `random_state` 和 `n_init`；
- 保存精确文件名和形状；
- 保存脚本 commit、随机种子和生成日志。

即使如此，自生成 anchors 也不应被称为“与作者严格一致”，除非结果文件校验一致。

---

## 7. 下载预训练权重

### 7.1 ResNet-50 ImageNet 权重

注意：配置路径是 `ckpts/`，不是 SparseDrive 文档中偶尔出现的 `ckpt/`。

```bash
mkdir -p ckpts

wget https://download.pytorch.org/models/resnet50-19c8e357.pth \
  -O ckpts/resnet50-19c8e357.pth
```

### 7.2 SparseDrive stage-1 checkpoint

```bash
wget \
  https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth \
  -O ckpts/sparsedrive_stage1.pth
```

官方 stage-2 配置明确：

```python
load_from = "ckpts/sparsedrive_stage1.pth"
```

这意味着论文 nuScenes 结果不是从随机初始化直接训练 DiffusionDrive，而是：

1. 先加载 SparseDrive stage-1；
2. 再训练 DiffusionDrive stage-2；
3. 检测、地图、运动和规划相关模块仍共同参与训练；
4. backbone 学习率使用主学习率的 0.1 倍。

### 7.3 DiffusionDrive 官方 nuScenes checkpoint

```bash
wget \
  https://huggingface.co/hustvl/DiffusionDrive/resolve/main/diffusiondrive_nusc_stage2.pth \
  -O ckpts/diffusiondrive_nusc_stage2.pth
```

检查：

```bash
ls -lh ckpts/
```

---

## 8. 阶段 A：先复现官方 checkpoint 的指标

### 8.1 官方评测命令

严格匹配作者文档：

```bash
bash tools/dist_test.sh \
  projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
  ckpts/diffusiondrive_nusc_stage2.pth \
  8 \
  --deterministic \
  --eval bbox
```

说明：

- `8` 表示 8 个 GPU process；
- 测试脚本默认 `--seed 0`；
- `--deterministic` 设置确定性随机种子和 cuDNN 选项；
- `--eval bbox` 是 MMDetection 测试入口要求的通用参数；
- 真正执行哪些任务由配置中的 `eval_mode` 决定；
- 该配置设置 `with_planning=True`，其余 det/tracking/map/motion 均为 False。

### 8.2 单 GPU 验证命令

只有 1 张 GPU 时可先做功能和指标验证：

```bash
bash tools/dist_test.sh \
  projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
  ckpts/diffusiondrive_nusc_stage2.pth \
  1 \
  --deterministic \
  --eval bbox
```

但由于推理过程包含随机高斯噪声：

- world size 改变会改变各 rank 的随机数序列；
- 单 GPU 与官方 8 GPU 日志可能存在轻微差异；
- 严格对齐官方数字时，应使用 8 GPU、seed 0 和相同软件栈；
- 不应把单 GPU 下最后几位不同立即判为实现错误。

### 8.3 预期输出

应看到类似：

```text
+-------------+--------+--------+--------+--------+--------+--------+--------+
| metrics     | 0.5s   | 1.0s   | 1.5s   | 2.0s   | 2.5s   | 3.0s   | avg   |
+-------------+--------+--------+--------+--------+--------+--------+--------+
| obj_box_col | 0.039% | 0.029% | 0.039% | 0.054% | 0.082% | 0.156% | 0.080% |
| L2          | 0.1688 | 0.2692 | 0.3918 | 0.5387 | 0.7096 | 0.9031 | 0.5704 |
+-------------+--------+--------+--------+--------+--------+--------+--------+
```

论文表格取其中：

```text
1 s: 0.2692 m / 0.029%
2 s: 0.5387 m / 0.054%
3 s: 0.9031 m / 0.156%
Avg: 0.5704 m / 0.080%
```

再四舍五入为：

```text
0.27 / 0.54 / 0.90 / 0.57
0.03 / 0.05 / 0.16 / 0.08
```

---

## 9. 必须理解 evaluator 如何计算指标

文件：

```text
projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py
```

### 9.1 时间采样

ego 规划轨迹：

```python
ego_fut_ts = 6
```

对应：

```text
0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s
```

### 9.2 L2 不是只取该时刻的 endpoint error

先计算每个 0.5 秒点上的 XY 欧氏距离：

\[
e_t = \left\|\hat{\mathbf p}_t-\mathbf p_t\right\|_2
\]

随后 evaluator 对每个 horizon 计算从第一个点到当前点的累计平均：

\[
L2_{1.0s}=\frac{e_{0.5s}+e_{1.0s}}{2}
\]

\[
L2_{2.0s}=\frac{e_{0.5s}+e_{1.0s}+e_{1.5s}+e_{2.0s}}{4}
\]

\[
L2_{3.0s}=\frac{1}{6}\sum_{t=0.5s}^{3.0s}e_t
\]

最后：

\[
L2_{\mathrm{Avg}}
=
\frac{
L2_{1.0s}+L2_{2.0s}+L2_{3.0s}
}{3}
\]

因此：

- 不能用纯 1 s / 2 s / 3 s endpoint distance 替代；
- 不能直接用常规 FDE；
- 不能用整条轨迹 ADE 后再重复填入三个 horizon；
- 更换 evaluator 后即使预测完全相同，也会得到不同表格。

### 9.3 碰撞指标使用 ego box

evaluator 构造的 ego box 尺寸为：

```text
length-like dimension: 4.084 m
width-like dimension: 1.85 m
height: 1.56 m
```

并按 UniAD 风格沿 heading 方向增加 0.5 m offset，再检测预测 ego polygon 与未来目标框 polygon 是否相交。

### 9.4 论文报告的是 `obj_box_col`

代码同时输出：

- `obj_col`：GT ego 轨迹本身与未来框发生碰撞的比例；
- `obj_box_col`：预测轨迹发生碰撞且 GT 轨迹没有碰撞的比例。

代码逻辑为：

```python
box_coll = pred_collision AND NOT gt_collision
```

论文表格中的 Collision 对应 `obj_box_col`，不是 `obj_col`。

### 9.5 碰撞率单位是百分数

内部值是比例，例如：

```text
0.0008
```

打印时乘 100，成为：

```text
0.080%
```

不要把 `0.0008` 写成 `0.0008%`，也不要把 `0.080%` 再乘一次 100。

### 9.6 不完整的未来轨迹会被跳过

代码中：

```python
if not sdc_planning_mask.all():
    continue
```

因此：

- 不是所有 val keyframe 都进入最终指标；
- scene 尾部缺少完整 3 秒 future GT 的样本会被排除；
- 修改过滤规则会改变分母；
- 对比其他论文时必须确认其是否采用同一过滤策略。

---

## 10. 阶段 B：完整重训练

### 10.1 官方核心训练参数

| 类别 | 参数 | 官方值 |
|---|---|---:|
| 数据版本 | `version` | `trainval` |
| train samples | `length["trainval"]` | 28130 |
| GPU 数 | `num_gpus` | 8 |
| 每 GPU batch | `batch_size` | 6 |
| global batch | `total_batch_size` | 48 |
| epochs | `num_epochs` | 10 |
| iter/epoch | `num_iters_per_epoch` | 586 |
| total iterations | `max_iters` | 5860 |
| optimizer |  | AdamW |
| 主学习率 | `lr` | 3e-4 |
| backbone LR multiplier |  | 0.1 |
| weight decay |  | 0.001 |
| gradient clipping |  | max norm 25 |
| LR schedule |  | CosineAnnealing |
| warmup |  | linear, 500 iters |
| warmup ratio |  | 1/3 |
| min LR ratio |  | 1e-3 |
| mixed precision |  | FP16, fixed loss scale 32 |
| 输入分辨率 |  | 704 × 256 |
| temporal queue | `queue_length` | 4 |
| ego future steps |  | 6 × 0.5 s = 3 s |
| agent future steps |  | 12 × 0.5 s = 6 s |
| ego modes / command |  | 6 |
| total plan anchors |  | 3 × 6 = 18 |
| cascade diffusion layers |  | 2 |
| diffusion inference steps |  | 2 |
| backbone |  | ResNet-50 |
| 初始化 |  | SparseDrive stage-1 |

计算：

```text
28130 // 48 = 586 iterations / epoch
586 × 10 = 5860 total optimizer iterations
```

由于整除向下取整，每个逻辑 epoch 不是完整遍历所有 28130 个样本；不要自行改为 `ceil` 后仍声称完全复现官方 recipe。

### 10.2 数据增强参数

```python
data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}
```

时序训练设置：

```python
with_seq_flag = True
sequences_split_num = 2
keep_consistent_seq_aug = True
queue_length = 4
```

不要让同一 temporal queue 内的图像使用彼此不一致的随机增强。

### 10.3 训练命令

```bash
export WORK_DIR="$PWD/work_dirs/diffusiondrive_nusc_stage2"
export GPUS=8
export CONFIG="projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"

mkdir -p "$WORK_DIR"

python -m torch.distributed.run \
  --nproc_per_node="${GPUS}" \
  --master_port=2333 \
  tools/train.py "${CONFIG}" \
  --launcher pytorch \
  --deterministic \
  --work-dir "${WORK_DIR}"
```

官方配置只在第 10 epoch，即第 5860 iteration 保存并评测一次：

```python
checkpoint_config.interval = 5860
evaluation.interval = 5860
```

因此不要期待每个 epoch 自动得到验证结果。

### 10.4 训练结束后独立评测

```bash
bash tools/dist_test.sh \
  projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
  work_dirs/diffusiondrive_nusc_stage2/iter_5860.pth \
  8 \
  --deterministic \
  --eval bbox
```

实际 checkpoint 名称以工作目录输出为准。

### 10.5 重训练的判断标准

优先级应为：

1. 官方 checkpoint 能否复现；
2. loss 是否稳定下降且无 NaN；
3. 最终 L2 / collision 是否接近官方；
4. 多随机种子结果的均值和波动；
5. 最后才比较日志中的最后几位小数。

官方只公开了一次训练日志，无法据此确定训练方差。因此，从头训练的单次结果略有波动是合理的；不要人为调 evaluator 来“对齐”数字。

---

## 11. nuScenes 版 truncated diffusion 的源码参数

论文描述了 truncated diffusion 的总体思想，但严格复现 nuScenes 结果时，应优先保留 `nusc` 分支的实际代码行为。

### 11.1 Scheduler

```python
DDIMScheduler(
    num_train_timesteps=1000,
    beta_schedule="scaled_linear",
    prediction_type="sample",
)
```

### 11.2 训练时的截断范围

nuScenes 代码训练时随机采样：

```python
timesteps = torch.randint(0, 40, ...)
```

即实际使用 `[0, 39]` 范围，而不是直接在完整 `[0, 999]` 上训练。

### 11.3 推理只执行 2 步

```python
step_num = 2
step_ratio = 40 / step_num
roll_timesteps = [20, 0]
```

模型先从 command-conditioned anchors 加入少量噪声：

```python
trunc_timesteps = 8
```

然后执行两次 scheduler step。

### 11.4 不要擅自“修正”这些魔数

nuScenes 代码中存在若干看起来不完全对称的设置：

- 训练 timestep 范围上限为 40；
- 初始 add-noise timestep 为 8；
- 两次反向 step 使用 `[20, 0]`；
- scheduler 的总训练步数仍为 1000。

这些是作者发布 checkpoint 所对应的实现。即使你认为可改进，也应先原样复现，再做消融实验。

---

## 12. 单张 RTX 4090 的现实方案

### 12.1 只验证官方 checkpoint

推荐直接用单 GPU 运行阶段 A。24 GB 显存通常足够。

### 12.2 单 GPU 从头训练不属于严格复现

官方配置为：

```text
8 GPUs × batch 6 = global batch 48
```

若简单改为：

```python
num_gpus = 1
total_batch_size = 6
batch_size = 6
```

则会同时改变：

- global batch；
- 每 epoch iteration 数；
- optimizer update 次数；
- warmup 相对位置；
- LR 与 batch 的匹配关系；
- 随机数序列；
- BatchNorm/数据分布行为；
- 总训练耗时。

可作为功能复现，但不能称为严格复现论文训练 recipe。

### 12.3 可选的单 GPU近似策略

#### 策略一：batch 6，按 10 个数据 epoch 训练

```text
per-GPU batch = 6
global batch = 6
iterations/epoch ≈ 28130 // 6
epochs = 10
```

可按线性规则把学习率作为起点缩放为：

\[
3\times10^{-4}\times\frac{6}{48}=3.75\times10^{-5}
\]

这只是工程启发式，不是作者配置。

#### 策略二：gradient accumulation 8 次

目标是保持 effective batch 48：

```text
micro batch 6 × accumulation 8 = effective batch 48
```

但必须保证：

- LR scheduler 按 optimizer update 而不是 micro-step 推进；
- warmup 仍为 500 optimizer updates；
- max iterations 的定义保持一致；
- FP16 hook 和 gradient clipping 的执行时机正确；
- checkpoint/evaluation interval 按 optimizer updates 计算。

MMCV 1.x 的 runner/hook 配置容易在这里产生“表面 batch 一致、实际 scheduler 不一致”的问题。除非专门验证过，不建议把该方案作为严格复现结果。

---

## 13. 常见失败模式与诊断

| 现象 | 高概率原因 | 处理 |
|---|---|---|
| 找不到 nuScenes 配置 | 还在 `main` 分支 | checkout `DiffusionDrive_nuScenes` 或 `nusc` |
| 找不到 `resnet50-19c8e357.pth` | 下载到了 `ckpt/` | 配置使用 `ckpts/` |
| 找不到 `sparsedrive_stage1.pth` | 文件位置不对 | 放入 `ckpts/` |
| 找不到 `kmeans_plan_6.npy` | 原脚本输出名不同 | 使用 release anchors |
| plan anchor shape 错误 | 把全局 K=6 当成 18 anchors | 使用 3 command × 6 mode 文件 |
| converter 报 CAN bus 缺失 | 未下载/解压 expansion | 确认 `data/nuscenes/can_bus/` |
| 训练时报 custom op 缺失 | 未编译 CUDA op | 执行 `python setup.py develop` |
| FlashAttention import/ABI 错误 | torch/CUDA/compiler 不匹配 | 对齐 torch1.13/cu116/GCC9.3 |
| `No module named diffusers` | 官方 requirement 漏项 | 安装并记录 diffusers |
| train sample 数不是 28130 | 数据版本/converter 不一致 | 停止训练并排查 |
| L2 明显比论文大 | evaluator、anchors、checkpoint 或坐标系错误 | 先用官方 ckpt 验证 |
| collision 大 100 倍 | 百分数单位处理错误 | 内部比例只乘一次 100 |
| 1 s L2 对不上 | 使用了 endpoint error | 使用累计平均实现 |
| test split 无法评测 | test 无未来真值 | 使用 `nuscenes_infos_val.pkl` |
| 单 GPU与官方末位不同 | RNG/world size 不同 | 用 8 GPU、seed0、deterministic |
| 改 GPU 数后训练不收敛 | global batch/LR/runner 改变 | 恢复官方 8×6 recipe |
| 只下载相机后训练失败 | 训练需 LiDAR depth supervision | 下载完整 trainval 传感器数据 |

---

## 14. 建议保存的复现实验记录

每次实验保存：

```text
repro/
├── git_commit.txt
├── environment_freeze.txt
├── nvidia_smi.txt
├── nvcc_version.txt
├── config_dump.py
├── train_info_summary.json
├── val_info_summary.json
├── anchor_shapes.txt
├── command.txt
├── train.log
├── eval.log
└── metrics.json
```

生成环境记录：

```bash
git rev-parse HEAD > repro/git_commit.txt
pip freeze > repro/environment_freeze.txt
nvidia-smi > repro/nvidia_smi.txt
nvcc --version > repro/nvcc_version.txt
```

保存命令：

```bash
printf '%q ' "$0" "$@" > repro/command.txt
```

建议额外记录：

- PyTorch、CUDA、cuDNN；
- GPU 型号和数量；
- train/val PKL 文件大小与 SHA256；
- 四个 anchor 文件 SHA256；
- stage-1 和 stage-2 checkpoint SHA256；
- seed；
- world size；
- 是否启用 deterministic；
- 实际 eligible validation sample 数；
- evaluator 源码 commit。

---

## 15. 最终验收清单

### 15.1 数据与文件

- [ ] 使用 nuScenes `v1.0-trainval`
- [ ] CAN bus expansion 已解压
- [ ] `nuscenes_infos_train.pkl` 为 28130 条
- [ ] val info 含未来 ego trajectory 与 `fut_boxes`
- [ ] 未使用 val 数据生成 anchors
- [ ] 使用作者 release 的四类 anchors
- [ ] `kmeans_plan_6.npy` 形状与模型一致
- [ ] ResNet-50 权重路径正确
- [ ] SparseDrive stage-1 权重路径正确
- [ ] DiffusionDrive stage-2 官方权重下载完整

### 15.2 环境

- [ ] Python 3.8.19
- [ ] PyTorch 1.13.0+cu116
- [ ] TorchVision 0.14.0+cu116
- [ ] MMCV 1.7.1
- [ ] MMDetection 2.28.2
- [ ] CUDA custom op 编译成功
- [ ] FlashAttention 可导入
- [ ] diffusers 已安装并记录版本

### 15.3 阶段 A

- [ ] 使用 `DiffusionDrive_nuScenes` release 或固定的 `nusc` commit
- [ ] test ann file 实际为 `nuscenes_infos_val.pkl`
- [ ] seed 为 0
- [ ] 使用 `--deterministic`
- [ ] evaluator 未修改
- [ ] L2 接近 `0.5704`
- [ ] `obj_box_col` 接近 `0.080%`

### 15.4 阶段 B

- [ ] 8 × GPU
- [ ] batch 6 / GPU
- [ ] global batch 48
- [ ] 10 epochs
- [ ] 586 iterations / epoch
- [ ] 5860 total iterations
- [ ] AdamW, LR 3e-4
- [ ] backbone LR multiplier 0.1
- [ ] warmup 500
- [ ] FP16 loss scale 32
- [ ] 从 SparseDrive stage-1 初始化
- [ ] 两层 cascade diffusion decoder
- [ ] 两步 diffusion inference
- [ ] 最终使用完全相同的 val evaluator

---

## 16. 建议的执行顺序

```text
1. checkout nuScenes release commit
2. 建立 Python 3.8 / torch1.13 / cu116 环境
3. 编译 custom CUDA op
4. 准备完整 nuScenes trainval + CAN bus
5. 生成 train/val PKL
6. 验证 train 样本数为 28130
7. 下载作者 release anchors
8. 验证 plan anchor 文件名和形状
9. 下载官方 stage-2 checkpoint
10. 在 val split 上跑阶段 A
11. 核对 0.5704 m / 0.080%
12. 下载 SparseDrive stage-1
13. 按 8×4090、global batch48 训练 5860 iterations
14. 独立运行同一评测命令
15. 保存环境、数据、anchors、checkpoint 和日志的哈希
16. 完成严格复现后再修改 diffusion、anchors 或训练设置
```

---

## 17. 官方资料

1. DiffusionDrive CVPR 2025 论文  
   https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html

2. DiffusionDrive GitHub  
   https://github.com/hustvl/DiffusionDrive

3. nuScenes 分支  
   https://github.com/hustvl/DiffusionDrive/tree/nusc

4. 官方训练与评测说明  
   https://github.com/hustvl/DiffusionDrive/blob/nusc/docs/train_eval.md

5. 官方 nuScenes 配置  
   https://github.com/hustvl/DiffusionDrive/blob/nusc/projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py

6. 官方 planning evaluator  
   https://github.com/hustvl/DiffusionDrive/blob/nusc/projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval.py

7. 官方 nuScenes release：日志和 clustered anchors  
   https://github.com/hustvl/DiffusionDrive/releases/tag/DiffusionDrive_nuScenes

8. 官方 nuScenes checkpoint  
   https://huggingface.co/hustvl/DiffusionDrive/blob/main/diffusiondrive_nusc_stage2.pth

9. SparseDrive Quick Start  
   https://github.com/swc-17/SparseDrive/blob/main/docs/quick_start.md

10. nuScenes devkit  
    https://github.com/nutonomy/nuscenes-devkit

---

## 18. 一句话结论

要复现论文中的 nuScenes 数字，最关键的不是只运行训练命令，而是同时固定：

```text
nusc release commit
+ v1.0-trainval 的官方 train/val split
+ 作者 release anchors
+ SparseDrive stage-1 初始化
+ 8×4090 / global batch48 / 5860 iterations
+ 两步 truncated diffusion 实现
+ 原始 planning_eval.py 的累计 L2 与 obj_box_col 定义
```

其中任何一项变化，都可能让结果不再与论文表格严格可比。
