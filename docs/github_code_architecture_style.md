# GitHub 代码架构样式（重构过程版）

本文定义“重构期间”GitHub 仓库应呈现的目录、分支、PR、发布样式，保证团队协作一致。

## 1. 仓库顶层样式（目标）

```text
programme/
├─ .github/
│  ├─ workflows/
│  │  └─ ci.yml
│  └─ PULL_REQUEST_TEMPLATE.md
├─ config/
├─ data/
├─ docs/
│  ├─ README.md
│  ├─ ros2_refactor_plan.md
│  └─ github_code_architecture_style.md
├─ examples/
├─ models/
├─ scripts/
├─ src/
├─ tests/
├─ ros2_ws/
│  ├─ src/
│  │  ├─ tactile_interfaces/
│  │  ├─ tactile_hardware/
│  │  ├─ tactile_perception/
│  │  ├─ tactile_control/
│  │  ├─ tactile_task/
│  │  ├─ tactile_ui_bridge/
│  │  └─ tactile_bringup/
│  └─ README.md
├─ main.py
├─ requirements.txt
└─ environment.yml
```

## 2. 分支样式

- `main`：线上稳定分支（必须可运行）
- `develop`：阶段集成分支
- `feature/<scope>-<name>`：功能分支
- `hotfix/<issue>`：紧急修复分支

示例：

- `feature/interfaces-msg-srv-action`
- `feature/hardware-learm-node`
- `feature/ui-bridge-readonly`

## 3. 提交样式

推荐格式：

- `feat(interfaces): add tactile raw/processed messages`
- `refactor(control): split gripper logic into ros2 service node`
- `docs(plan): add phase-2 rollback checklist`

要求：

- 每个 commit 可独立回滚。
- 不混合“重命名 + 大逻辑改动 + 文档大改”到单一 commit。

## 4. PR 样式

每个 PR 必须回答四个问题：

1. 这次重构了哪些模块？
2. 原有功能是否保留？如何验证？
3. 回滚方式是什么？
4. 是否影响目录/接口，文档是否同步？

PR 大小建议：

- 单 PR 改动文件 <= 30（尽量）
- 单 PR 聚焦一个目标（如“仅硬件节点化”）

## 5. Release 样式

重构过程建议双轨标签：

- 里程碑标签：`phaseX-*`
- 产品版本标签：`vX.Y.Z`

示例：

- `phase1-mvp`
- `phase2-hardware`
- `v0.2.0`

Release Notes 统一结构：

1. 本版目标
2. 主要变更
3. 保留能力
4. 已知问题
5. 回滚指引

## 6. 阶段内目录演进样式

### 阶段 1 后

- 新增：`ros2_ws/src/tactile_interfaces`
- 新增：`ros2_ws/src/tactile_bringup`
- 新增：`ros2_ws/src/tactile_ui_bridge`（只读）

### 阶段 2 后

- 新增：`ros2_ws/src/tactile_hardware`
- 旧 `src/core/hardware_interface.py` 仍保留

### 阶段 3 后

- 新增：`ros2_ws/src/tactile_control`
- 旧 `control_thread` 路径仍可运行

### 阶段 4 后

- GUI 默认仍可 legacy，支持 `ros2` 模式切换

### 阶段 5 后

- 新增：`ros2_ws/src/tactile_task`
- demo 流程分批迁移，不一次替换

### 阶段 6 后

- 补齐联调文档、回滚清单、发布资产

## 7. 文件一致性规则

- 目录命名：小写下划线，禁空格
- 行尾：LF（由 `.gitattributes` 控制）
- 编码：UTF-8（由 `.editorconfig` 控制）
- 大文件：模型与日志分离管理，日志不入库
- 接口变更必须同步 `docs/` 与示例

## 8. 同步节奏样式（团队默认）

- 每天开始：`pull --rebase`
- 功能完成：push + 开 PR
- PR 合并：全员同步 pull
- 阶段收尾：打 tag + 发 release
