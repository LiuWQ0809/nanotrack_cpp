# NanoTrack C++ (ROS 2)

本项目是 NanoTrack 视觉跟踪算法的高性能 ROS 2 C++ 移植版本，专门针对 NVIDIA Jetson 平台利用 TensorRT 和 CUDA 进行了优化。

## 功能特性

- **全 C++ 实现**：核心跟踪逻辑完全使用 C++ 重写，以实现最高性能。
- **TensorRT 推理**：使用 TensorRT 引擎进行骨干网络（backbone）和头部网络（head）的推理加速。
- **CUDA 预处理**：自定义 CUDA 核函数进行图像裁剪、缩放和归一化（去除了 OpenCV 的运行时依赖）。
- **ROS 2 集成**：标准的 ROS 2 节点接口。
- **无 GUI 依赖**（核心）：可视化效果直接渲染到图像话题上，支持通过 `rqt_image_view` 或内置的 Python 可视化工具进行远程调试。
- **稳定跟踪**：实现了卡尔曼滤波（Kalman Filter）和位置平滑算法，有效防止边界框抖动。

## 依赖项

- ROS 2 (Humble/Foxy)
- CUDA (11.4+)
- TensorRT (8.x)
- OpenCV (仅核心模块，用于内部数据处理，非必须)

## 目录结构

- `src/`: C++ 源代码 (`nanotrack.cpp`, `tracker_node.cpp`, `preprocessor.cu`)
- `include/`: 头文件
- `scripts/`: 辅助脚本，包含用于桌面交互的 `visualizer_node.py`
- `launch/`: Launch 启动文件
- `models/`: 存放 TensorRT 引擎文件 (`.engine`) 的目录

## 使用方法

### 1. 编译 (Build)
使用提供的脚本编译工作空间。
```bash
./compile.sh
```

### 2. 运行 (Run)
启动跟踪器和可视化节点。
```bash
./run.sh
```
这将启动：
- `nanotrack_tracker`：主要的 C++ 跟踪节点。
- `visualizer_node.py`：一个 Python 节点，用于订阅可视化话题并弹出交互窗口。

### 3. 交互 (Interaction)
- 将会出现一个标题为 **"NanoTrack C++ Visualizer"** 的窗口。
- 使用 **鼠标左键** 在想要跟踪的物体周围画一个框。
- 跟踪器将立即初始化并开始跟踪该物体。

### 4. 停止 (Stop)
安全停止所有相关节点：
```bash
./stop.sh
```

##话题 (Topics)

- **输入图像**: `/cr/camera/bgr/front_left_960_768` (可在 `launch/nanotrack_tracker.launch.py` 中配置)
- **输出可视化**: `/nanotrack/visualization`
- **ROI 控制**: `/target_roi` (在此话题发布 `sensor_msgs/RegionOfInterest` 可通过程序启动跟踪)

## 配置 (Configuration)

修改 `launch/nanotrack_tracker.launch.py` 以更改参数：
- `engine_dir`：包含 `.engine` 文件的目录。
- `image_topic`：输入相机话题。
- `min_confidence`：置信度阈值（默认 `0.6`）。
- `track_lost_threshold`：目标丢失判定的阈值（默认 `0.4`）。
