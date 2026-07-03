# 05. 可视化指南

## Camera mosaic

```bash
python scripts/visualize_sample.py --mode camera-mosaic --sample-index 0 --with-boxes
```

输出 6 相机环视图。`--classes car pedestrian truck` 可以只显示指定 detection 类别或原始类别。

## LiDAR BEV

```bash
python scripts/visualize_lidar_bev.py --sample-index 0 --with-boxes --color-by distance
```

`--color-by intensity` 可按强度上色。`--nsweeps 10` 会调用官方 devkit 的多 sweep 聚合逻辑。

## Radar BEV

```bash
python scripts/visualize_radar_bev.py --sample-index 0 --radar all --velocity compensated
```

点颜色默认表示 RCS，箭头表示速度。`--disable-filters` 会关闭官方 Radar 默认过滤，适合调试原始 PCD 字段。

## LiDAR / Radar 投影到 Camera

```bash
python scripts/project_lidar_to_camera.py --sample-index 0 --camera CAM_FRONT
python scripts/project_radar_to_camera.py --sample-index 0 --camera CAM_FRONT --radar all --color-by velocity
```

这两个脚本体现完整的跨时间戳坐标转换链路。

## Scene 轨迹

```bash
python scripts/visualize_scene.py --scene-name scene-0061 --show-map
```

当前实现会稳定输出 ego trajectory。地图 raster/vector 叠加依赖 map expansion 文件，后续可在 `visualization/maps.py` 中扩展。

## Scene 级别同步视频

```bash
python scripts/visualize_scene_video.py \
  --version v1.0-mini \
  --scene-name scene-0061 \
  --out outputs/scene-0061_sixcam_lidar_bev.mp4 \
  --preview-out outputs/scene-0061_preview.png
```

画面结构：

```text
上半部分: CAM_FRONT_LEFT / CAM_FRONT / CAM_FRONT_RIGHT
          CAM_BACK_LEFT  / CAM_BACK  / CAM_BACK_RIGHT

下半部分: 黑色背景 LIDAR_TOP BEV + sample_annotation 3D boxes
```

下方 BEV 会保持米制等比例缩放并居中显示，车头方向朝上；黑色 3D 面板可以占满下半部分，
但点云和标注框本身不会被横向拉伸。需要调整可视范围时使用 `--bev-xlim` 和 `--bev-ylim`。

常用参数：

```bash
--fps 2.0                 # nuScenes keyframe 标注为 2Hz
--max-frames 10           # 快速预览
--nsweeps 5               # BEV 聚合多帧 LiDAR，速度会变慢
--classes car pedestrian  # 只显示部分类别
--no-boxes                # 不显示标签框
--width 1920 --height 1080
--codec XVID --out outputs/video.avi
```

该脚本按 scene 逐帧流式生成，兼容 `v1.0-mini`、`v1.0-trainval` 和 `v1.0-test`。`v1.0-test` 没有公开标注，视频中会显示传感器画面和 LiDAR 点云，但不会有 annotation boxes。
