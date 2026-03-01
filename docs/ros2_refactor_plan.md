# ROS2 渐进式重构计划（含 GitHub 同步策略）

## 1. 目标与边界

### 1.1 目标

- 保留现有 PyQt 界面与主要算法能力。
- 先把通信与控制链路 ROS2 化，再逐步迁移任务编排。
- 架构按分层解耦：展示层、业务层、设备层、基础层。
- 支持后续接入 MoveIt2 与多机协同。

### 1.2 不变项（必须保留）

- `main.py` 可继续运行（legacy 模式）。
- 现有 GUI 外观与交互不重写。
- 每阶段都可回滚到上一稳定 tag。

### 1.3 双轨运行策略

- `--mode legacy|ros2`（默认 `legacy`）。
- 新能力先并行上线，再灰度切流，最后替换。

## 2. 目标包结构

- `tactile_interfaces`
- `tactile_hardware`
- `tactile_perception`
- `tactile_control`
- `tactile_task`
- `tactile_ui_bridge`
- `tactile_bringup`

建议落地路径：

- `ros2_ws/src/<packages>`

## 3. 通信设计基线

### 3.1 Topics

- `/tactile/raw`
- `/tactile/processed`
- `/arm/state`
- `/gripper/state`
- `/system/health`

### 3.2 Services

- `/arm/enable`
- `/arm/home`
- `/gripper/set_force`
- `/system/reset_emergency`

### 3.3 Actions

- `/task/execute_demo`
- `/arm/move_joints`
- `/gripper/grasp_sequence`

### 3.4 QoS 建议

- 传感器类：`SensorDataQoS`（高频、BestEffort、KeepLast）
- 状态类：Reliable
- 控制命令：Reliable
- 监控类：可 BestEffort

## 4. 分阶段实施（每次重构什么 + 是否保留原功能）

### 阶段 1（1 周）

重构内容：

- 建立 `ros2_ws` 与 `tactile_interfaces`。
- 打通最小链路：假数据发布 + GUI 订阅（只读）。
- 增加 `tactile_bringup` 最小 launch。

保留情况：

- legacy 功能 100% 保留。

验收：

- `colcon build` 通过。
- GUI 可实时显示 `/tactile/raw`。
- legacy 启动链路不受影响。

### 阶段 2（1-2 周）

重构内容：

- 把 `hardware_interface`、`learm_interface` 封装为硬件节点（建议 LifecycleNode）。
- 发布硬件状态与健康信息。

保留情况：

- 旧硬件路径继续可用，切换由启动参数控制。

验收：

- 节点可独立启停、断连恢复、状态稳定发布。

### 阶段 3（1-2 周）

重构内容：

- `control_thread`、`gripper_controller` 拆为控制节点 + 服务接口。
- 接入安全监控与急停复位链路。

保留情况：

- 旧控制线程保留，作为 fallback。

验收：

- 关键控制路径可通过 Service/Action 完整闭环。

### 阶段 4（1 周）

重构内容：

- GUI 接入 `tactile_ui_bridge`（先读后写）。
- 替换数据源与命令出口，不改界面布局。

保留情况：

- UI 代码预计保留约 80%。
- 保留 legacy 与 ros2 两种运行模式。

验收：

- 同一 UI 可切换 `legacy`/`ros2`。
- 关键交互不退化。

### 阶段 5（1-2 周）

重构内容：

- `demo_manager` 迁移为 Action 协调器（`tactile_task`）。
- 逐个迁移演示流程（非一次性迁移）。

保留情况：

- 未迁移 demo 保留旧入口。

验收：

- 至少两个核心 demo 在 ROS2 模式可稳定运行。

### 阶段 6（1 周）

重构内容：

- 联调、压测、录包回放、故障演练。
- 固化运维文档、发布流程与回滚预案。

保留情况：

- legacy 模式保留到 ROS2 稳定两个迭代周期后再评估下线。

验收：

- 发布候选版通过联调清单并完成 release。

## 5. GitHub 同步与版本策略（关键）

### 5.1 分支策略

- `main`：始终可运行、可演示。
- `develop`：阶段集成。
- `feature/*`：单功能开发。
- `hotfix/*`：线上修复。

### 5.2 每天什么时候 pull

- 开始开发前：`pull --rebase origin develop`
- 提交前：再 `pull --rebase` 一次（减少冲突）
- PR 合并后：所有人立即 pull 同步
- 发布前：`main` 再 pull 并做最终校验

### 5.3 什么时候 push 到 GitHub

- 本地完成一个可自测的小功能即 push（不攒大提交）。
- 每日至少一次同步 push。
- 所有里程碑必须通过 PR 合并，不直推 `main`。

### 5.4 什么时候发 Release

- 每个阶段结束发一个里程碑版本（tag + notes）。
- 例如：`phase1-mvp`, `phase2-hardware`, `v1.0.0-ros2`。

## 6. 功能保留与回滚机制（关键）

### 6.1 功能保留承诺

- 每次迭代都要证明 legacy 可运行。
- 新功能先挂在 ros2 模式，不覆盖 legacy 主流程。

### 6.2 回滚机制

- 任何异常可立即切回 `--mode legacy`。
- 以阶段 tag 为回滚点，支持快速回退。
- 禁止跨阶段的大爆炸重构。

## 7. 文件管理与 GitHub 一致性（关键）

### 7.1 目录规范

- 业务代码：`src/`
- ROS2 工程：`ros2_ws/`
- 文档：`docs/`
- 示例与脚本：`examples/`, `scripts/`
- 测试：`tests/`

### 7.2 命名规范

- 全小写 + 下划线
- 禁止空格目录名
- 接口、参数、launch 命名统一

### 7.3 变更约束

- 一次提交只做一类变更（代码/配置/文档分离）。
- 变更接口必须同步文档与示例。
- PR 必须包含验证结果与回滚说明。

## 8. CI 与质量门禁

- PR 必过：
- 代码风格检查
- 基础语法检查
- 关键模块导入/构建检查
- 文档完整性检查（接口变更必须有文档）

建议后续补充：

- ROS2 包 `colcon build` 检查
- 关键节点 smoke test

## 9. 环境建议

- 主开发/联调：Ubuntu 24.04 + ROS2 Jazzy
- Windows：用于轻量开发、GUI 逻辑开发
- 真机联调建议 Linux 统一环境

## 10. 执行节奏建议

- 每周一：阶段目标与验收项更新到 GitHub Milestone
- 每天：小步提交 + PR
- 每周：阶段集成回归 + 里程碑 tag/release
- 每阶段结束：输出一份“保留功能清单 + 回滚验证记录”
