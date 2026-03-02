# ROS2 渐进式重构总计划（含 SmolVLA 策略迁移）

本文档是当前项目的唯一重构主计划，覆盖架构、分阶段实施、GitHub 同步策略、回滚策略和验收标准。

## 1. 重构目标

### 1.1 业务目标

- 保留现有 PyQt GUI 的外观和主要交互流程。
- 把通信与控制链路从单体线程模型迁移到 ROS2 节点模型。
- 支持后续 MoveIt2、多机协同和录包回放。
- 将原自研 AI 组件迁移为开源 VLA 方案（优先 SmolVLA），并保留可回退能力。

### 1.2 工程目标

- 每一步可回滚，不影响当前可运行版本。
- 采用双轨运行：`legacy` 与 `ros2` 并行。
- GitHub 作为唯一真源，所有里程碑通过 PR/Tag/Release 固化。

### 1.3 非目标（当前阶段不做）

- 不一次性重写 GUI。
- 不一次性替换所有控制逻辑。
- 不在早期阶段删除 legacy 主链路。

## 2. 目标架构（重构完成态）

### 2.1 分层架构

- 展示层：PyQt HMI（通过 `tactile_ui_bridge` 读写 ROS2）
- 业务层：任务编排与策略（Action/Service 驱动）
- 设备层：传感器、机械臂、夹爪驱动节点
- 基础层：`interfaces`、`launch`、参数、日志、录包

### 2.2 逻辑数据流

```text
Hardware Nodes -> /tactile/raw -> Perception -> /tactile/processed
                                         -> Policy (SmolVLA/Legacy)
Policy -> /policy/command -> Control -> /arm/* /gripper/*
Control/Hardware -> /system/health -> UI Bridge -> PyQt GUI
Task Action Server <-> UI Bridge / external clients
```

### 2.3 包结构（最终）

- `tactile_interfaces`
- `tactile_hardware`
- `tactile_perception`
- `tactile_control`
- `tactile_task`
- `tactile_ui_bridge`
- `tactile_bringup`
- `tactile_policy`（新增：策略后端抽象，承载 SmolVLA）

建议目录：

- `ros2_ws/src/<above_packages>`

## 3. SmolVLA 接入策略（新增）

### 3.1 原则

- 不直接硬替换控制器，先替换“策略层”。
- 控制与安全仍由 `tactile_control` 执行和限幅。
- 通过配置切换后端，随时回退：
- `policy_backend=legacy|smolvla`

### 3.2 新增组件

- `tactile_policy` 包
- `policy_server` 节点（可本地推理或远程推理）
- `policy_adapter`（输入输出对齐：视觉/触觉/状态 -> 动作建议）

### 3.3 输入输出约定

- 输入：
- `/tactile/processed`
- `/arm/state`
- `/gripper/state`
- 可选 `/vision/features`
- 输出：
- `/policy/command`（动作建议，不直接下发硬件）
- `/policy/status`（推理延迟、置信度、后端状态）

### 3.4 安全约束

- 控制层对策略输出做限幅、速率限制、急停仲裁。
- 策略超时/异常自动回退到 `legacy` 策略。

## 4. 通信设计（基线）

### 4.1 Topics

- `/tactile/raw`
- `/tactile/processed`
- `/arm/state`
- `/gripper/state`
- `/system/health`
- `/policy/command`（新增）
- `/policy/status`（新增）

### 4.2 Services

- `/arm/enable`
- `/arm/home`
- `/gripper/set_force`
- `/system/reset_emergency`
- `/policy/set_backend`（新增）

### 4.3 Actions

- `/task/execute_demo`
- `/arm/move_joints`
- `/gripper/grasp_sequence`

### 4.4 QoS 规范

- 传感器流：BestEffort + KeepLast（高频）
- 状态流：Reliable
- 控制命令：Reliable
- 告警/健康：Reliable（必要时 TransientLocal）

## 5. 分阶段计划（每次改什么、保留什么、验收什么）

### Phase 0：冻结基线（已完成）

改动：

- 建立 `main` 稳定线和 `develop` 集成线。
- 固化文档、PR 模板、CI 基础校验。

保留：

- 全量 legacy 可运行。

验收：

- 基线 release/tag 可回滚。

### Phase 1：最小链路打通（进行中）

改动：

- `ros2_ws`、`tactile_interfaces`、`tactile_bringup`、`tactile_ui_bridge` 骨架。
- 假数据发布 + UI 订阅。
- 新增 `main_ros2.py`（只读监控模式）。

保留：

- `main.py` legacy 100% 保留，不替换。

验收：

- `colcon build` 通过。
- `/tactile/raw` 可观测。
- GUI 在 ROS2 入口可显示流数据。
- legacy 入口不受影响。

### Phase 2：硬件层节点化

改动：

- `hardware_interface`、`learm_interface` 节点化到 `tactile_hardware`。
- 输出标准状态和健康信号。

保留：

- 旧硬件路径可切回。

验收：

- 节点可独立启停，断连恢复有效。

### Phase 3：控制层迁移

改动：

- `control_thread`、`gripper_controller` 迁移到 `tactile_control`。
- Service/Action 闭环。

保留：

- 旧控制线程保留为 fallback。

验收：

- 关键控制流程可在 ROS2 下稳定执行。

### Phase 4：GUI 桥接切换

改动：

- GUI 数据源与命令出口切到 `tactile_ui_bridge`。
- UI 保持形态不变。

保留：

- 支持 `legacy|ros2` 运行模式切换。

验收：

- 常用交互无回归，显示性能达标。

### Phase 5：任务编排迁移

改动：

- `demo_manager` 迁移到 `tactile_task` Action 协调器。

保留：

- 未迁移 demo 仍走旧入口。

验收：

- 至少两个核心 demo 在 ROS2 模式可稳定运行。

### Phase 6：SmolVLA 接入（新增关键阶段）

改动：

- 新增 `tactile_policy`。
- 实现 `legacy_policy` 与 `smolvla_policy` 双后端。
- 控制层接收策略建议并执行安全仲裁。

保留：

- 默认可继续使用 `legacy_policy`。

验收：

- 可动态切换后端。
- SmolVLA 推理链路稳定，异常自动回退。
- 无硬件损伤风险（安全限幅有效）。

### Phase 7：联调与发布固化

改动：

- 压测、故障演练、录包回放。
- 文档补齐与发布流程固化。

保留：

- legacy 至少保留两个迭代周期再评估下线。

验收：

- 达成发布门槛并发布 `v1.0.0-ros2`（或等价版本）。

## 6. 功能保留与退场策略

### 6.1 保留策略

- 所有阶段必须验证 legacy 可运行。
- 新链路默认不开启硬切换，先灰度。

### 6.2 legacy 下线条件

- ROS2 + 新策略链路功能对等。
- 连续两个迭代周期稳定通过。
- 关键故障演练通过。
- 已打 `legacy-final` 回滚点。

## 7. GitHub 同步策略（关键）

### 7.1 分支模型

- `main`：始终可运行
- `develop`：阶段集成
- `feature/*`：单功能分支
- `hotfix/*`：线上修复

### 7.2 pull 规则（固定时机）

- 每天开工前：`git pull --rebase origin develop`
- 每次提交前：再 `pull --rebase` 一次
- PR 合并后：所有成员立即 pull
- 发布前：`main` 再次 pull 与核验

### 7.3 push 规则

- 完成一个可自测小目标即 push。
- 禁止长时间本地堆积大改动。
- 禁止直推 `main`。

### 7.4 PR 与 Release

- 每阶段 2-5 个小 PR，不做“大爆炸合并”。
- 每阶段结束打 tag 并发 release notes。

## 8. 文件管理与仓库一致性

### 8.1 目录规范

- 业务代码：`src/`
- ROS2：`ros2_ws/`
- 文档：`docs/`
- 脚本/示例：`scripts/`、`examples/`
- 测试：`tests/`

### 8.2 命名和格式

- 小写下划线命名，禁空格目录。
- UTF-8 + LF。
- 日志、构建产物不入库。

### 8.3 文档一致性要求

- 接口、话题、服务变更必须同步 `docs/`。
- 策略后端变更必须同步“回退说明”。

## 9. 风险与缓解

### 9.1 高风险点

- Qt 事件循环与 ROS2 执行器并发模型。
- 策略输出不稳定导致控制风险。
- 真机联调环境漂移（不同机器依赖不一致）。

### 9.2 缓解

- 先只读桥接，后控制写入。
- 策略层与控制层强隔离，控制层最终裁决。
- Linux 统一联调环境（Ubuntu 24.04 + ROS2 Jazzy）。

## 10. 阶段完成定义（DoD）

- 代码合并：PR 全绿、审查通过。
- 运行验证：按阶段验收清单执行并留记录。
- 文档完整：计划、接口、回滚说明同步。
- 可回滚：有明确 tag 和回退步骤。
