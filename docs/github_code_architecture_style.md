# GitHub 代码架构样式（重构执行版）

本文件规定重构期间 GitHub 仓库应该长期保持的结构和协作样式。

## 1. 目标目录样式

```text
programme/
├─ .github/
│  ├─ workflows/ci.yml
│  └─ PULL_REQUEST_TEMPLATE.md
├─ config/
├─ data/
├─ docs/
│  ├─ README.md
│  ├─ ros2_refactor_plan.md
│  ├─ github_code_architecture_style.md
│  └─ phase1_kickoff.md
├─ examples/
├─ models/
├─ scripts/
├─ src/
├─ tests/
├─ ros2_ws/
│  ├─ README.md
│  └─ src/
│     ├─ tactile_interfaces/
│     ├─ tactile_hardware/
│     ├─ tactile_perception/
│     ├─ tactile_control/
│     ├─ tactile_task/
│     ├─ tactile_ui_bridge/
│     ├─ tactile_bringup/
│     └─ tactile_policy/
├─ main.py
├─ main_ros2.py
├─ requirements.txt
└─ environment.yml
```

## 2. 分支与里程碑样式

- `main`：稳定可运行分支
- `develop`：阶段集成分支
- `feature/<scope>-<name>`：功能分支
- `hotfix/<issue>`：热修复分支

里程碑命名：

- `phase1-*`
- `phase2-*`
- ...
- 产品版本：`vX.Y.Z`

## 3. 提交样式（Commit）

推荐：

- `feat(interfaces): add tactile raw message`
- `feat(policy): add smolvla backend skeleton`
- `refactor(control): migrate gripper command flow to service`
- `docs(plan): update phase gates and rollback criteria`

约束：

- 一次提交只做一类改动（代码/配置/文档分离）。
- 每个提交可独立回滚。

## 4. PR 样式（必须回答）

每个 PR 描述必须包含：

1. 这次改动了哪些模块。
2. 旧功能是否保留，如何验证。
3. 回滚方式与回滚点。
4. 是否更新了文档和配置。

建议：

- 单 PR 聚焦一个目标，不混杂多个阶段任务。
- 单 PR 文件改动尽量可审查（优先小步）。

## 5. Release 样式

发布说明固定结构：

1. 本版目标
2. 主要变更
3. 保留能力（legacy/回退）
4. 已知问题
5. 回滚指引

## 6. 重构阶段目录演进样式

### Phase 1

- 新增 `tactile_interfaces`、`tactile_bringup`、`tactile_ui_bridge`
- 新增 `main_ros2.py`（只读监控入口）

### Phase 2-5

- 逐步补齐 `tactile_hardware`、`tactile_control`、`tactile_task`

### Phase 6（策略迁移）

- 新增 `tactile_policy`
- 提供 `legacy_policy` 与 `smolvla_policy`
- 通过参数切换后端，不硬删旧策略

## 7. 与 GitHub 一致性的硬约束

- 命名统一：小写下划线，禁空格路径。
- 文本统一：UTF-8 + LF。
- 构建产物不入库：`build/`, `install/`, `log/`, 缓存目录。
- 接口变化必须同步文档和示例。

## 8. 每日协作节奏（团队默认）

- 开工：`pull --rebase`
- 功能完成：push + 开 PR
- PR 合并：全员 pull
- 阶段收尾：tag + release

## 9. AI 组件迁移样式（新增）

- 原自研 AI 不直接删除，先置为 `legacy_policy` 后端。
- SmolVLA 接入在 `tactile_policy`，输出策略建议，不直连硬件。
- 控制层保留最终裁决和安全限幅。
- 满足稳定门槛后再考虑 legacy AI 退场。
