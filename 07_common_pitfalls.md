# 07. 常见坑

## 1. dataroot 指错

`--dataroot` 指向 nuScenes 根目录，不是 `v1.0-mini` 内部。

```text
正确: ../nuscenes
错误: ../nuscenes/v1.0-mini
```

## 2. 解压覆盖

full trainval/test 会有多个 archive。解压时应合并到同一个 `nuscenes/` 根目录，不要覆盖已有 `samples/`、`sweeps/`、`maps/`。

## 3. sample 和 sample_data 混淆

`sample` 是 2Hz 标注关键帧，`sample_data` 是具体传感器文件。annotation 挂在 sample 上，不挂在单个 camera 或 LiDAR 文件上。

## 4. global box 直接投影

annotation box 默认在 global frame。要画到相机图像上，必须：

```text
global box -> ego frame at camera timestamp -> camera frame -> image plane
```

## 5. Radar 速度字段

Radar 同时有 `vx/vy` 和 `vx_comp/vy_comp`。做感知特征时通常优先使用补偿后的 `vx_comp/vy_comp`。

## 6. test set 没有公开标注

test set 不能本地计算 detection 指标，只能提交 JSON 到官方评估服务器。

## 7. 原始类别和 detection 类别不同

原始类别约 23 类，detection benchmark 是 10 类。训练检测模型时要做类别映射和过滤。

## 8. lidarseg category 覆盖

如果额外下载 lidarseg，相关 `category.json` 可能替换 full dataset 的 category 文件。它通常不影响 3D object detection，但文档和实验记录里要写清楚。

## 9. BEV 中 LiDAR 点云和 box 坐标系不一致

原始 `LIDAR_TOP` 点云在 LiDAR 传感器坐标系；`sample_annotation` 的
`translation/rotation` 在 global 坐标系。如果把 box 只转到 ego frame，
再叠加到原始 LiDAR 点云上，就会出现标注框和点云错位。

正确做法是让二者使用同一个参考系：

- 要么把 LiDAR 点云转到 ego frame，同时 box 也转到 ego frame；
- 要么保持 LiDAR 点云在 `LIDAR_TOP` frame，同时把 box 转到 `LIDAR_TOP` frame。

本工程采用第二种，和官方 `nusc.get_sample_data()` 的默认行为一致。
