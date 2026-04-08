# ROS 2 环境与依赖安装指南

这份文档面向两类人：

- 第一次在 WSL2 / Ubuntu 24.04 上搭这套项目的人
- 已经装过 Conda，但现在要把当前 ROS 2 Web 主链真正跑起来的人

当前推荐基线：

- WSL2
- Ubuntu 24.04
- ROS 2 Jazzy
- 当前主链入口：`ros2_ws/src/tactile_bringup/launch/tactile_grasp_studio.launch.py`

ROS 2 Jazzy 的基础安装步骤参考官方文档：

- [ROS 2 Jazzy Ubuntu Debian 安装文档](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

## 1. 先分清 Conda 环境和 ROS 2 环境

这个仓库现在同时保留两套环境，但用途不一样：

- 根目录 `environment.yml`
  - 保留的 Conda 环境
  - 主要给旧桌面端、训练/分析脚本、历史 Python-only 工具链使用
- `/opt/ros/jazzy` + `ros2_ws/`
  - 当前 Web 主链真正运行的环境
  - 机械臂仿真、MoveIt、Gazebo、Web 网关、视觉主链都走这里

保留的 Conda 环境仍然建议继续保留，更新方式：

```bash
cd /home/whispers/programme
conda env update -f environment.yml --prune
conda activate dayiprogramme312
```

注意：当前 Web 主链不要只靠这个 Conda 环境运行，它不是当前 ROS 2 主运行环境。

## 2. 在 Ubuntu 24.04 上安装 ROS 2 Jazzy

下面这组命令对应官方 Jazzy Debian 安装路径，适合新的 Ubuntu 24.04 机器。

### 2.1 设置 locale

```bash
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale
```

### 2.2 打开 Universe 仓库并添加 ROS 软件源

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
sudo apt install -y curl
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | head -n 1 | cut -d'"' -f4)
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

### 2.3 安装 ROS 基础工具和 Jazzy

这个项目建议直接装 `desktop`，因为当前链路会用到 RViz、MoveIt、Gazebo 调试能力。

```bash
sudo apt update
sudo apt install -y ros-dev-tools ros-jazzy-desktop
```

把 ROS 环境写进 shell：

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

## 3. 初始化 rosdep

如果这台机器从来没有初始化过 rosdep，先执行：

```bash
sudo rosdep init
rosdep update
```

这一步只需要做一次。

## 4. 安装这个项目额外需要的系统依赖

ROS 2 Jazzy 装好以后，还需要补当前项目的系统依赖。当前主链至少要装这些：

```bash
sudo apt update
sudo apt install -y \
  python3-pip \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-fastapi \
  python3-uvicorn \
  python3-websockets \
  python3-requests \
  ros-jazzy-moveit \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-trajectory-controller \
  ros-jazzy-rmw-cyclonedds-cpp \
  ros-jazzy-xacro \
  npm
```

如果你要用 RealSense 真机相机链路，再补：

```bash
sudo apt install -y ros-jazzy-realsense2-camera
```

## 5. 安装当前主链的非 ROS Python 依赖

当前视觉和 Web 网关代码除了 ROS 包之外，还会直接 import 这些 Python 包：

- `fastapi`
- `uvicorn`
- `websockets`
- `requests`
- `ultralytics`
- `onnx`
- `onnxruntime`
- `open3d`

仓库里已经单独补了一份文件：

- [`ros2_ws/requirements-ros2.txt`](../ros2_ws/requirements-ros2.txt)

安装命令：

```bash
python3 -m pip install --user --break-system-packages \
  -r /home/whispers/programme/ros2_ws/requirements-ros2.txt
```

说明：

- `open3d` 在当前主链里是增强项，主要用于点云滤波和离群点剔除，强烈建议安装
- `detector_seg_node` 当前默认会走 ONNX 路径，所以 `onnx` 和 `onnxruntime` 不建议省略

## 6. 构建 ROS 2 工作区

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 7. 构建 Web 前端

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm install
npm run build
```

如果你在开发前端，也可以单独开：

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm run dev
```

## 8. 配置对话 / 语义模型

当前主链会读取：

```text
~/.config/programme/remote_vlm.env
```

最小示例：

```bash
mkdir -p ~/.config/programme
cat > ~/.config/programme/remote_vlm.env <<'EOF'
PROGRAMME_DIALOG_MODEL_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
PROGRAMME_DIALOG_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
PROGRAMME_DIALOG_API_KEY=EMPTY
EOF
```

## 9. 启动当前全链路

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
source install/setup.bash
ros2 launch tactile_bringup tactile_grasp_studio.launch.py
```

默认 Web 地址：

```text
http://127.0.0.1:8765
```

## 10. 额外说明

### 10.1 GraspGen 不是 ROS 2 工作区内部依赖

当前主抓取候选后端配置为 `graspgen_zmq`，默认会连：

- `127.0.0.1:5556`

所以你还需要单独把 GraspGen server 启起来。

### 10.2 为什么还要保留 `environment.yml`

因为仓库里仍然保留了旧 `src/`、旧桌面端和一些 Python-only 工具。你现在做主链调试主要用 ROS 2 shell，但那份 Conda 环境对历史代码和数据处理脚本依然有价值，所以这里不删，只做补全。
