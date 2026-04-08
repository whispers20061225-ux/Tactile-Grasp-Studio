# ROS2 Workspace Quickstart

这个 `ros2_ws/` 是当前 Tactile Grasp Studio Web 主链真正运行的 ROS 2 工作区。

如果你是在新机器上第一次搭环境，先看：

- [ROS 2 环境与依赖安装指南](../docs/ros2_environment_setup_guide.md)
- [仓库根 README](../README.md)

## 1. 先分清两套环境

- `/opt/ros/jazzy` + 当前 `ros2_ws/`：现在真正跑 Web 抓取主链的运行环境
- `../environment.yml`：保留的 Conda 环境，用于旧 Python 工具链、旧桌面端和一些辅助脚本
- [`requirements-ros2.txt`](requirements-ros2.txt)：当前 ROS 2 shell 里还要补装的非 ROS Python 包

当前主链不要直接依赖 Windows 侧 Python，也不要把根目录 Conda 环境当作唯一运行环境。

## 2. 安装当前工作区依赖

先确保 ROS 2 Jazzy 已经装好并且能 `source /opt/ros/jazzy/setup.bash`。

安装系统依赖：

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

如果你要走真实相机链路，再装：

```bash
sudo apt install -y ros-jazzy-realsense2-camera
```

安装当前主链会 import 的非 ROS Python 包：

```bash
python3 -m pip install --user --break-system-packages \
  -r /home/whispers/programme/ros2_ws/requirements-ros2.txt
```

## 3. 构建工作区

第一次在这台机器上使用 rosdep 时：

```bash
sudo rosdep init
rosdep update
```

然后构建：

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## 4. 构建 Web 前端

`tactile_web_gateway` 会直接托管 `frontend/dist`，所以第一次跑主链前先打包：

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm install
npm run build
```

如果你是在改前端，也可以跑开发服务器：

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm run dev
```

## 5. 配置对话 / 语义模型

当前主链默认读取：

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

## 6. 启动当前主链

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
source install/setup.bash
ros2 launch tactile_bringup tactile_grasp_studio.launch.py
```

默认 Web 入口：

```text
http://127.0.0.1:8765
```

## 7. 当前主链关键节点

- `qwen_semantic_node`
- `detector_seg_node`
- `cloud_filter_node`
- `primitive_fit_node`
- `grasp_input_cloud_node`
- `grasp_backend_node`
- `task_executive_node`
- `search_target_skill_node`
- `sim_pick_task_node`
- `tactile_web_gateway`

## 8. 快速自检

检查节点：

```bash
ros2 node list | grep -E "tactile_web_gateway|qwen_semantic_node|detector_seg_node|cloud_filter_node|primitive_fit_node|grasp_backend_node|task_executive_node|sim_pick_task_node"
```

检查网关：

```bash
curl http://127.0.0.1:8765/api/bootstrap
```

检查前端产物：

```bash
test -f /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend/dist/index.html
```
