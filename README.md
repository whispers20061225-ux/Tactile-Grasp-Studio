# Tactile Grasp Studio: 用自然语言控制机器人抓取

> 就像跟助手说话一样简单。你只要说一句“抓那个蓝色圆柱”，系统就会去看、去理解、去规划，然后把它抓起来。

Tactile Grasp Studio 是一个桌面机器人抓取系统，但它的使用方式并不“机器人”。你**不需要懂 ROS、不用写代码、也不用记命令**。打开网页，输入一句自然语言，就可以让系统把“看懂目标、锁定目标、规划抓取、执行动作、回传状态”这一整条链路跑起来。

你只需要三步：
1. **打开网页** — `http://127.0.0.1:8765`
2. **用中文说想抓什么** — “抓取蓝色圆柱体”、“抓右边那个杯子”
3. **看着机器人执行** — 从识别、锁定、规划到抓取，全过程实时可见

就这么直接。

如果你是第一次接触机器人，它足够好上手；如果你是开发者、调试人员，或者要做方案演示的人，它也足够能打。因为它不只是“能抓”，而是把“听懂你要什么、看懂场景里哪个是目标、规划怎么抓、执行时发生了什么”全都放到了同一个界面里。

## 🎯 为什么选它？

### 对新手友好
- **零门槛** — 不用学 ROS 话题，不用理解节点和 launch，会打字、会点网页就能开始
- **自然语言交互** — 支持中英文混合输入，“抓那个蓝色的东西”“抓左边那个杯子”这种说法也能懂
- **不是黑盒** — 它不会让你“点一下然后干等着”，当前识别到了什么、锁定了哪个目标、准备怎么抓，都能直接看到
- **Web 界面** — 打开浏览器就能操作，不用装一堆客户端，也不用来回切窗口

### 对开发者友好
- **全过程可见** — 识别、锁定、几何处理、候选抓取、执行状态，每一步都有可视化反馈
- **调试友好** — 视觉、触觉、日志、控制都在一个页面里，不用一边盯终端一边猜系统在干什么
- **模块化设计** — 语义理解、目标检测、点云处理、抓取策略、执行控制都是解耦的，替换模型和算法成本低
- **接口清晰** — 各层职责明确，既方便定位问题，也方便后续扩展成你自己的系统

### 对演示友好
- **完整链路** — 从“听懂一句话”到“真正抓起来”，不是拼接几个孤立模块，而是一条完整闭环
- **实时状态** — 当前在做什么、下一步准备做什么、卡在了哪里，页面上都能看得很清楚
- **一键复位** — `Reset Scene` 可以快速回到初始状态，适合连续展示和反复试验
- **说服力强** — 展示的不只是一个抓取动作，而是一整套“理解任务并完成任务”的能力

### 技术实力
- **多模态语义理解** — 不只是识别物体类别，而是能结合语言和画面理解“哪个”“左边那个”“蓝色那个”这类指代
- **开放词汇检测与分割** — 不被固定类别表绑死，面对没专门训练过的目标，也能给出可用候选
- **3D 几何推理** — 从 2D 检测反投影到 3D 点云，再做滤波、聚类、几何补全，让抓取不是“看起来像对了”，而是“位置真的对了”
- **智能抓取规划** — GraspGen 生成候选，MoveIt 2 负责 pregrasp / grasp 轨迹规划，既能看见候选，也能真正落到执行
- **触觉反馈闭环** — 不只是机械臂动起来，真实触觉还可以参与反馈，让抓取更接近真实可落地的操作链路
- **一体化可视化** — 控制、视觉、触觉、日志、状态机信息都集中在 Web 页面里，既能用，也能讲，也能查

## 🚀 快速体验

```bash
# 1. 启动系统
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
source install/setup.bash
ros2 launch tactile_bringup programme_system.launch.py

# 2. 打开浏览器
# http://127.0.0.1:8765

# 3. 输入任务
# "抓取蓝色圆柱体"
# "抓右边那个蓝色圆柱"
# "只用平行夹爪抓取蓝色圆柱"
```

---

> **当前推荐运行环境**:WSL2 / Ubuntu 24.04 / ROS 2 Jazzy / Web 主链
> **历史兼容**:旧的 `main.py` + Qt UI 仍保留,但不再是当前主链路

## 📚 文档导航

第一次接触?建议先看**术语表**,再按模块读说明书。

| 文档 | 适合谁看 |
|------|----------|
| [术语与缩写表](docs/terminology_guide.md) | 所有人,先搞懂名词 |
| [系统总览说明书](docs/system_overview_manual.md) | 想了解整体架构 |
| [Web 控制台说明书](docs/web_console_manual.md) | 日常使用者 |
| [语义感知说明书](docs/semantic_perception_manual.md) | 想改视觉/语义模块 |
| [点云与几何说明书](docs/pointcloud_geometry_manual.md) | 想改点云处理 |
| [抓取规划与执行说明书](docs/grasp_execution_manual.md) | 想改抓取策略 |
| [触觉与运维说明书](docs/tactile_operations_manual.md) | 运维/调试人员 |
| [文档总索引](docs/README.md) | 完整文档列表 |
| [ROS 2 环境安装指南](docs/ros2_environment_setup_guide.md) | 新机器配置 |

---

## 🏗️ 系统架构

### 核心链路

```
自然语言指令 → 多模态语义理解 → 开放词汇分割与 ROI 锁定 →
深度反投影与点云滤波 → 几何拟合/补全 → 抓取候选生成 →
pregrasp/grasp 规划 → 执行 → Web 端监控与复位
```

### 入口选择

| Launch 文件 | 用途 |
|-------------|------|
| `programme_system.launch.py` | **推荐**:完整系统(视觉 + 触觉 + 抓取) |
| `web_console_stack.launch.py` | 轻量模式:只启动 Web 界面 |
| `tactile_grasp_studio.launch.py` | 兼容旧命令 → 转发到 `programme_system.launch.py` |
| `programme_mainline.launch.py` | 兼容旧命令 → 转发到 `programme_system.launch.py` |

### 核心能力

- ✅ 中英文任务输入,自动结构化为 `SemanticTask`
- ✅ 基于语义提示的开放词汇实例分割与目标锁定
- ✅ 2D 检测 → 3D 点云反投影,滤波/平面剔除/聚类/几何补全
- ✅ GraspGen 生成抓取候选 + MoveIt 2 规划 pregrasp/grasp
- ✅ Web 端四合一:`Control` / `Vision` / `Tactile` / `Logs`
- ✅ 快捷操作:`Re-plan` / `Return Home` / `Reset Scene` + 调试视图

---

## 📦 核心模块

| 包 | 作用 |
| --- | --- |
| `tactile_bringup` | 🚀 Launch 和参数总入口 |
| `tactile_web_bridge` | 🌐 FastAPI/WebSocket 网关 + React/Vite 前端 |
| `tactile_vision` | 👁️ 语义理解、开放词汇分割、点云处理、几何拟合 |
| `tactile_task` | 🧠 搜索与任务编排 |
| `tactile_task_cpp` | 🦾 MoveIt 执行、pregrasp/grasp 规划、return home |
| `tactile_control` | 🛡️ 控制层安全代理 |
| `tactile_hardware` | 🔌 机械臂与触觉传感器驱动（含仿真 fallback） |
| `tactile_sim` | 🎮 Gazebo 仿真、场景复位、搜索扫描 |

> 💡 **新手提示**：理解系统请从 `tactile_bringup` 和 `tactile_web_bridge` 开始，不要先看旧的 `src/`、`main.py`、PyQt。

## 3. 运行环境建议

### 3.1 推荐运行环境

- WSL2
- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Sim / MoveIt 2
- Node.js + npm(用于构建前端)

当前这份 README 以实际运行仓库 `/home/whispers/programme` 为准。

### 3.2 关于 Python 环境

这个仓库同时保留了两套历史:

- `environment.yml` / `requirements.txt`:保留的 Conda/Python 工具链,用于旧桌面端、训练/分析脚本和一些独立调试工具
- `ros2_ws/`:当前 Web 主链对应的 ROS 2 工作区

当前 Web 主链建议在 WSL 的 ROS 2 shell 中运行,不要把 Windows 侧 Python 环境当作当前主运行环境。

如果你是新机器第一次配置,建议直接按这两个入口走:

- [ROS 2 环境与依赖安装指南](docs/ros2_environment_setup_guide.md)
- [ROS2 工作区 Quickstart](ros2_ws/README.md)

保留的 Conda 环境仍然可以继续用,更新命令如下:

```bash
cd /home/whispers/programme
conda env update -f environment.yml --prune
conda activate dayiprogramme312
```

### 3.3 系统依赖

至少需要这些系统级依赖:

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
  ros-jazzy-xacro \
  ros-jazzy-rmw-cyclonedds-cpp \
  npm
```

如果你要走真实相机链路,再额外安装:

```bash
sudo apt install -y ros-jazzy-realsense2-camera
```

当前 Web 主链的非 ROS Python 依赖建议直接按这个文件装:

```bash
python3 -m pip install --user --break-system-packages \
  -r /home/whispers/programme/ros2_ws/requirements-ros2.txt
```

`open3d` 在当前主链里是可选但强烈建议的增强项,主要用于更稳的点云滤波和离群点剔除。

## 4. 当前主链启动前置条件

### 4.1 构建 ROS 2 工作区

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
sudo rosdep init  # 仅第一次在这台机器上需要
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

### 4.2 构建 Web 前端

`tactile_web_gateway` 默认会直接托管 `frontend/dist`,所以第一次运行前要先构建:

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm install
npm run build
```

如果你在改前端,也可以在开发模式下单独跑:

```bash
cd /home/whispers/programme/ros2_ws/src/tactile_web_bridge/frontend
npm run dev
```

### 4.3 配置多模态语义模型

当前主链会从 `~/.config/programme/remote_vlm.env` 读取对话/语义模型配置。一个最小例子:

```bash
mkdir -p ~/.config/programme
cat > ~/.config/programme/remote_vlm.env <<'EOF'
PROGRAMME_DIALOG_MODEL_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions
PROGRAMME_DIALOG_MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct-AWQ
PROGRAMME_DIALOG_API_KEY=EMPTY
EOF
```

如果你用的是 DashScope 或其他 OpenAI-compatible 服务,也可以填:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_MODEL`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

### 4.4 启动 GraspGen ZMQ 服务

当前主抓取候选后端配置为 `graspgen_zmq`,默认连接:

- host: `127.0.0.1`
- port: `5556`
- repo: `/home/whispers/GraspGen`

先确保 GraspGen 服务已经运行。按本机现有 GraspGen 仓库的 client-server 文档,一个常用启动方式是:

```bash
cd /home/whispers/GraspGen
source .venv-cu128/bin/activate
pip install pyzmq msgpack msgpack-numpy
python client-server/graspgen_server.py \
  --gripper_config /path/to/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml \
  --port 5556
```

如果你已经有自己的 GraspGen server 管理方式,只要保证 `127.0.0.1:5556` 可用即可。

### 4.5 启动对话/VLM 服务

README 不强绑定具体服务实现;你只需要保证上面 `remote_vlm.env` 里的 endpoint 已可访问。
当前代码默认把它当作 OpenAI-compatible 的 chat completion 接口来调用。

## 5. 启动当前全链路

### 5.1 推荐启动命令

```bash
source /opt/ros/jazzy/setup.bash
cd /home/whispers/programme/ros2_ws
source install/setup.bash
ros2 launch tactile_bringup programme_system.launch.py
```

这条 launch 会拉起当前主链路中的关键节点,包括:

- `realsense2_camera_node`(或对应相机驱动链)
- `qwen_semantic_node`
- `detector_seg_node`
- `cloud_filter_node`
- `primitive_fit_node`
- `grasp_input_cloud_node`
- `grasp_backend_node`
- `stm32_bridge_node`
- `grasp_profile_node`
- `task_executive_node`
- `search_target_skill_node`
- `sim_pick_task_node`
- `tactile_web_gateway`

默认情况下,`programme_system.launch.py` 和它的兼容别名不会再拉起 `tactile_sim_node`,避免真实触觉和仿真触觉同时发布到 `/tactile/raw`。如果你需要轻量 Web 入口,请使用 `web_console_stack.launch.py`。

### 5.2 打开 Web 页面

默认地址:

```text
http://127.0.0.1:8765
```

默认网关参数来自 `tactile_web_gateway`:

- host: `127.0.0.1`
- port: `8765`

## 6. 推荐操作流程

1. 打开 `Control` 页,先确认后端连通。
2. 在对话框里输入任务,例如:
   - `抓取蓝色圆柱体`
   - `抓右边那个蓝色圆柱`
   - `只用平行夹爪抓取蓝色圆柱`
3. 先用 `Review` 模式检查结构化任务是否正确;确认后再 `Execute`。
4. 到 `Vision` 页确认目标框、候选列表和当前锁定实例是否正确。
5. 抓取结束后,优先使用 `Reset Scene` 回到初始状态。

当前配置里,`Reset Scene` 会在重置场景后自动 `Return Home`。

## 7. 快速自检

### 7.1 看节点是否都起来了

```bash
ros2 node list | grep -E "tactile_web_gateway|stm32_bridge_node|grasp_profile_node|qwen_semantic_node|detector_seg_node|cloud_filter_node|primitive_fit_node|grasp_backend_node|task_executive_node|sim_pick_task_node"
```

### 7.2 看 Web 网关是否可响应

```bash
curl http://127.0.0.1:8765/api/bootstrap
```

### 7.3 看语义与视觉是否有输出

```bash
ros2 topic echo /qwen/semantic_task --once
ros2 topic echo /perception/detection_result --once
ros2 topic echo /grasp/candidate_grasp_proposals --once
```

## 8. 现在该看哪份文档

- 第一次看这些英文缩写:先看 [术语与缩写表](docs/terminology_guide.md)
- 当前系统的手册首页:[系统总览说明书](docs/system_overview_manual.md)
- Web 交互入口:[Web 控制台说明书](docs/web_console_manual.md)
- 语义与目标锁定:[语义感知说明书](docs/semantic_perception_manual.md)
- 点云、反投影和拟合:[点云与几何说明书](docs/pointcloud_geometry_manual.md)
- 抓取候选与执行规划:[抓取规划与执行说明书](docs/grasp_execution_manual.md)
- 触觉与测试复位:[触觉与运维说明书](docs/tactile_operations_manual.md)
- 文档总索引:[docs/README.md](docs/README.md)
- Windows/VM 和 RealSense 的历史部署文档:
  - [windows_ros2_realsense_quickstart.md](docs/windows_ros2_realsense_quickstart.md)
  - [windows_vm_one_click_runbook.md](docs/windows_vm_one_click_runbook.md)
  - [windows_vm_split_phaseA.md](docs/windows_vm_split_phaseA.md)
  - [windows_vm_split_phaseB.md](docs/windows_vm_split_phaseB.md)

## 9. 说明

- 本 README 只覆盖当前 Tactile Grasp Studio Web 主链路。
- 旧 Qt UI 不再作为当前推荐入口。
- `main.py`、PyQt、旧 `src/` 目录仍保留,但它们主要属于历史兼容路径。
- 阶段化命名现在只在历史 kickoff 文档中保留,当前默认入口统一使用职责化正式命名。
