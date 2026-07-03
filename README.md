# nuscenes_learn

一个面向 nuScenes mini/full 数据集的本地学习工程：先跑通数据检查、schema 查询和可视化，再理解 3D detection 训练与评估流程。

服务器上建议目录结构如下，本工程已经按这个布局设置默认路径：

```text
parent_dir/
├── nuscenes_learn
└── nuscenes
    ├── maps
    ├── samples
    ├── sweeps
    └── v1.0-mini
```

如果你在 `nuscenes_learn` 目录内运行脚本，默认 `--dataroot` 是 `../nuscenes`。也可以显式传入其他路径。

## 1. 在服务器已有环境中运行

```bash
cd nuscenes_learn
conda activate hjl_env
python scripts/check_dataset.py --version v1.0-mini
```

你的 ModelArts/NPU 环境如果已经有 `nuscenes-devkit`、`numpy`、`matplotlib`、`Pillow` 等包，就不需要执行 `pip install -r requirements.txt`。本工程的脚本会自动把 `src/` 加入 Python 路径，也不要求 `pip install -e .`。

如果你确实想把本工程作为包安装到当前环境，建议只安装本项目本身，不触发依赖升级：

```bash
pip install -e . --no-deps
```

`requirements.txt` 只作为全新干净环境的兜底参考，不建议在已经配置好的 NPU 环境中直接执行。

## 2. 检查数据集

```bash
python scripts/check_dataset.py --version v1.0-mini
```

或指定路径：

```bash
python scripts/check_dataset.py --dataroot ../nuscenes --version v1.0-mini
```

## 3. 查看 scene 和 sample

```bash
python scripts/list_scenes.py --version v1.0-mini
python scripts/inspect_sample.py --version v1.0-mini --sample-index 0
```

## 4. 可视化一个样本

```bash
python scripts/visualize_sample.py --version v1.0-mini --sample-index 0 --mode camera-mosaic --with-boxes
python scripts/visualize_lidar_bev.py --version v1.0-mini --sample-index 0 --with-boxes
python scripts/visualize_radar_bev.py --version v1.0-mini --sample-index 0 --radar all
python scripts/project_lidar_to_camera.py --version v1.0-mini --sample-index 0 --camera CAM_FRONT
python scripts/project_radar_to_camera.py --version v1.0-mini --sample-index 0 --camera CAM_FRONT --radar all
```

默认输出写入 `outputs/`，不会修改 nuScenes 原始数据。

## 5. 生成 scene 级别同步视频

生成类似 nuScenes 可视化界面的 mp4：上方 6 路相机画面，下方黑色 LiDAR BEV 点云和 3D 标签框。

```bash
python scripts/visualize_scene_video.py \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --out outputs/scene-0061_sixcam_lidar_bev.mp4 \
  --preview-out outputs/scene-0061_preview.png
```

完整 trainval/test 数据集也用同一个入口，只需换 `--version` 和 `--scene-name`。脚本按 scene 的 sample 顺序逐帧读取和写入视频，不会一次性加载整套数据。

```bash
python scripts/visualize_scene_video.py \
  --version v1.0-trainval \
  --scene-name scene-0001 \
  --max-frames 40
```

如果服务器 mp4 编码不可用，可以改成 avi：

```bash
python scripts/visualize_scene_video.py --version v1.0-mini --scene-name scene-0061 --codec XVID --out outputs/scene-0061.avi
```
服务器上用法：
cd nuscenes_learn
conda activate hjl_env

python scripts/visualize_scene_video.py \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --out outputs/scene-0061_sixcam_lidar_bev.mp4 \
  --preview-out outputs/scene-0061_preview.png

效果是：
    上半部分：6 路相机，2 x 3 排列，并叠加 3D box 和类别标签
    下半部分：黑色 3D 面板中居中显示纵向 LiDAR BEV，车头朝上，点云和标签框不横向拉伸
    按 scene 的 sample 顺序逐帧写视频，默认 fps=2.0
    兼容 v1.0-mini、v1.0-trainval、v1.0-test
    v1.0-test 没公开标注，所以不会画 annotation boxes
    
如果 mp4 编码不可用：
python scripts/visualize_scene_video.py \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --codec XVID \
  --out outputs/scene-0061.avi

## 6. 推荐学习顺序

1. 读 `docs/00_overview.md` 和 `docs/01_dataset_structure.md`，确认目录与 13 张 JSON 表。
2. 跑 `check_dataset.py`、`list_scenes.py`、`inspect_sample.py`，理解 token 关系。
3. 跑 camera/LiDAR/Radar 可视化脚本。
4. 读 `docs/04_coordinate_transforms.md`，对照 `project_lidar_to_camera.py` 看坐标链路。
5. 读 `docs/06_detection_training_and_eval.md`，理解检测 10 类、NDS/mAP 和提交 JSON。

## 7. 常见路径错误

`--dataroot` 应该指向 nuScenes 根目录，而不是版本目录：

```text
正确: ../nuscenes
错误: ../nuscenes/v1.0-mini
```
