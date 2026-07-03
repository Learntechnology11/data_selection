# 04. 坐标系与转换

nuScenes 里最容易出错的是坐标系。核心链路是：

```text
sensor frame
 -> ego frame at that sensor timestamp
 -> global frame
 -> ego frame at target sensor timestamp
 -> target sensor frame
 -> camera intrinsic projection
 -> image pixel
```

## 坐标系

| 坐标系 | 含义 |
| --- | --- |
| sensor frame | 某个 camera/LiDAR/Radar 自己的坐标系 |
| ego frame | 自车坐标系，随车辆移动 |
| global frame | log 地图坐标系 |
| camera pixel frame | 图像像素坐标 |
| BEV frame | 俯视图，通常 x forward、y left |
| box frame | 3D box 自身局部坐标系 |

## 投影伪代码

```python
p_global = T_global_ego_src @ T_ego_src_sensor @ p_sensor
p_ego_cam = inverse(T_global_ego_cam) @ p_global
p_cam = inverse(T_ego_cam_sensor) @ p_ego_cam
uv_depth = K @ p_cam[:3]
u = uv_depth[0] / uv_depth[2]
v = uv_depth[1] / uv_depth[2]
```

注意：

1. 只保留 `depth > 0` 的点。
2. 只保留落在图像范围内的点。
3. annotation box 默认在 global frame，不能直接用 camera intrinsic 投影。
4. 四元数顺序是 `w, x, y, z`。

本工程对应实现见 `src/nusc_lab/transforms.py`，投影脚本见：

```bash
python scripts/project_lidar_to_camera.py --sample-index 0 --camera CAM_FRONT
python scripts/project_radar_to_camera.py --sample-index 0 --camera CAM_FRONT --radar all
```

## BEV 坐标一致性

LiDAR `.bin` 文件中的点默认位于 `LIDAR_TOP` 传感器坐标系，而 `sample_annotation`
中的 box 默认位于 global 坐标系。绘制 LiDAR BEV 时不能只把 box 转到 ego
frame 后直接叠加到原始点云上，否则会出现点云和标注框整体平移或旋转不一致。

本工程的 BEV 绘制遵循 nuScenes devkit 的做法：

```text
global annotation box
 -> ego frame at LIDAR_TOP timestamp
 -> LIDAR_TOP sensor frame
 -> BEV x/y plot
```

对应实现是 `nusc.get_sample_data(sample["data"]["LIDAR_TOP"], selected_anntokens=...)`。
官方 devkit 会返回已经变换到当前 `sample_data` 传感器坐标系下的 boxes，因此可以和
原始 `LIDAR_TOP` 点云直接叠加。

## Ego Trajectory 坐标系

`ego_pose.translation` 表示某个 `sample_data` 时间戳下，自车在该 log 的 global
map frame 中的位置。绘制 scene 级 `ego_trajectory` 时，本工程默认使用每个 sample
的 `LIDAR_TOP` keyframe sample_data：

```text
sample
 -> sample["data"]["LIDAR_TOP"]
 -> sample_data["ego_pose_token"]
 -> ego_pose["translation"][:2]
 -> global map xy trajectory
```

因此 `ego_trajectory.png` 的横纵轴是 global x/y 米制坐标，不是以第一帧为原点的
ego/local 坐标，也不是 LiDAR 传感器坐标。后续如果叠加官方 map expansion，必须继续
使用同一 location 的 global map 坐标；如果要画“以当前车为中心”的局部轨迹，需要额外做
`global -> current ego` 变换。
