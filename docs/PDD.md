# 优化后的 UI 布局建议（面向 PySide6 实现）

本文件目标是将“可复现、可视化、交互式调参、重训练/逐日模拟”工作流以工程化方式落地，并把要在 UI 中暴露的参数（LightGBM / 神经网络 / PPO / 映射器 / 交叉验证 / 特征工程 等）直接映射为可实现的控件与信号流程。

文档为前端工程师或开发团队提供建议：包括推荐的 PySide6 组件、信号/槽设计、后台执行模式、产物保存策略与校验逻辑，便于实现和交付。

---

## 目录

1. 总体架构与主窗口
2. 顶部控制栏（快速控制）
3. 左侧面板：数据与特征工程（参数表单）
4. 中央主区：选项卡（训练 / 诊断 / 解释 / 回测）
5. 右侧面板：关键指标与运行对比
6. 底部区域：运行历史、日志与产物管理
7. 弹窗 / 确认 / 高耗任务 UX 规范
8. 后端交互与并发模型（PySide6 实现要点）
9. 配置 / 预设 / 导入导出 / 可复现性（JSON / MLflow）
10. 校验规则与自动诊断（重要约束）
11. 视觉与交互细节（快捷键、无障碍、图表选择）
12. 文件与产物组织约定

---

## 1. 总体架构与主窗口

推荐主窗体为 `QMainWindow`：

- 中央 widget 使用水平主布局 `QHBoxLayout`，并用 `QSplitter` 分为三列：
  - 左侧：`QScrollArea`，包含参数树或表单（`QVBoxLayout` / `QFormLayout` / `QTreeWidget`），建议占比约 20%
  - 中间：`QTabWidget`，包含训练、诊断、解释与回测面板，建议占比约 55%
  - 右侧：`QWidget`，显示关键指标与运行对比，建议占比约 25%
- 顶部：`QToolBar` 或自定义横栏，放置快速控件与小 KPI 卡片
- 底部：`QSplitter` 下方为运行历史与日志控制台（使用 `QTableView` + `QPlainTextEdit`）

建议持久化 `QSplitter` 布局，方便用户拖拽并保存布局偏好。

---

## 2. 顶部控制栏（快速控制）

推荐将以下控件放在顶部工具栏或自定义横栏：

- `QComboBox`：模型选择（LightGBM / LSTM / MLP / PPO / Ensemble），切换时通过 `QStackedWidget` 显示对应参数面板
- 运行按钮 `QPushButton`（主动作，绿色）与停止按钮 `QPushButton`（次要动作，红色）
  - 运行触发 `start_run(config)` 信号
  - 停止触发 `stop_run()` 信号（实现应为安全中断请求）
- 预设下拉 `QComboBox` + 应用按钮
- 随机种子 `QSpinBox` 与可复现开关 `QCheckBox`（控制是否保存环境哈希等复现信息）
- 运行模式切换（快速运行 / 完整训练）：使用 `QRadioButton` 或 `QComboBox`
- KPI 卡片区（自定义 widget）展示小图与关键指标：如最近一次调整后夏普、年化收益、年化波动、平均换手等（可用 `QWebEngineView` 嵌入 Plotly 或使用 `FigureCanvasQTAgg`）

行为建议：

- 运行时禁用部分配置控件并显示进度 spinner（`QProgressDialog` 或自定义进度条），但不要显示不可靠的 ETA；显示折次/轮次状态与文本描述
- 在触发运行前执行轻量快速的校验（详见第 10 节）

---

## 3. 左侧面板：数据与特征工程（参数表单）

左侧建议使用 `QFormLayout` / `QTreeWidget` 或可折叠的 `QGroupBox`（模拟 Accordion）组织参数。以下为各分组与推荐控件：

数据处理
- 文件路径：`QLineEdit` + `QPushButton`（打开文件对话框，选择 `train.csv` / `test.csv`）
- 缺失值策略：`QComboBox`（如 ffill / bfill / median / zero / linear）
- 是否添加 `_is_missing` 指示：`QCheckBox`
- 特征群组多选：`QListWidget`（带复选框）或一组 `QCheckBox`
- 滞后与滚动窗口配置：`QLineEdit`（逗号分隔）或多选控件
- 目标特征数：`QSpinBox`（50–500，建议默认 375）
- 是否使用 SHAP：`QCheckBox` 与 SHAP 采样大小 `QSpinBox`

序列验证
- CV 方法：`QComboBox`（Purged K-Fold / Walk-Forward）
- 若为 Purged K-Fold：`QSpinBox` n_splits、`QSpinBox` embargo_days、`QSpinBox` label_horizon
- 若为 Walk-Forward：`QSpinBox` train_window、`QSpinBox` val_window、`QSpinBox` step
- 验证按钮：`QPushButton`（触发快速样本校验）

模型设置
·LightGBM 参数（仅当模型为 LightGBM 时显示）
- 学习率：`QDoubleSpinBox`
- 叶子数：`QSpinBox`
- 最小子样本数：`QSpinBox`
- 特征子采样 / bagging 比例：`QDoubleSpinBox`
- 迭代次数与早停：`QSpinBox`
- 是否启用 Optuna 以及试验次数：`QCheckBox` + `QSpinBox`

·神经网络参数（LSTM / MLP）
- 网络类型选择：`QComboBox`（LSTM / MLP）
- LSTM 相关：序列长度、hidden_dim、层数、dropout（使用 `QSpinBox` / `QDoubleSpinBox`）
- 训练设置：预训练/微调轮数、学习率、批大小、换手正则等
- 双精度几何均值开关：`QCheckBox`

·PPO 参数
- 窗口长度 / 回合长度 / 换手成本：`QSpinBox` / `QDoubleSpinBox`
- PPO 超参：学习率、更新次数、每次更新步数、轮数、小批量大小、截断系数等

映射 / 后处理
- 映射类型：`QComboBox`（线性 / Tanh / 自定义）
- 自定义时：提供若干 `QDoubleSpinBox` 供参数调节，并可打开 CMA-ES 模态
- 是否应用 Isotonic 校准以及越界处理选项

预设与保存
- 预设下拉 + 保存/加载配置按钮（使用文件对话框导出/导入 JSON）

UX 建议：每个控件提供 `setToolTip()` 和合理的 min/max/step 校验。

---

## 4. 中央主区：选项卡（训练 / 诊断 / 解释 / 回测）

建议使用 `QTabWidget` 提供四个主选项卡：

Tab A — 训练（Run）
- 左上显示当前运行配置摘要（只读）
- 右上放置运行控制（Run / Stop / 快速运行切换 / 预设应用）
- 中间为实时日志 `QPlainTextEdit`（只读，接收后台日志信号）
- 下方为 CV 表格（`QTableView`），展示每折指标与产物路径

行为：运行触发后禁用部分控件，后台通过 `progress_update(str)` 等信号推送日志与状态；运行完成后展示 summary（均值±标准差），并支持将当前运行标记为 baseline

Tab B — 诊断（Diagnostics）
- 使用 Plotly（嵌入 `QWebEngineView`）展示交互式图表，也可用 Matplotlib 导出高分辨率图片
- 包含：仓位随时间图、策略 vs 市场累计收益、滚动 Sharpe、换手统计、映射预览等

Tab C — 解释（Explain，SHAP & 特征）
- SHAP 汇总图（前 30）与时间演化 SHAP，提供局部与全局解释功能
- 计算 SHAP 为异步任务并限制样本量，UI 上显示进度与估算消耗

Tab D — 回测（Backtest / Rolling Simulation）
- 提供重训练频率选择、warmup 天数、回测结果面板（净值曲线、回撤、月度热力图）与交易明细表

---

## 5. 右侧面板：关键指标与运行对比

右侧纵向布局包含：

- KPI 卡片组：展示 adjusted_sharpe、年化收益、波动率、换手、最大回撤等（每项配小火花图）
- Top-Features 列表：展示前 10 特征与趋势（SHAP）
- 运行对比选择区：可多选 2–3 个运行并点击 Compare 生成对比图表
- 产物快速链接：列出模型、isotonic pickle、CMA-ES 参数等并支持下载

---

## 6. 底部区域：运行历史、日志与产物管理

使用 `QTabWidget` 划分 History / Artifacts / Console：

- History：`QTableView` 显示 run_id、时间戳、模型、配置摘要、调整后夏普与状态，可筛选并右键 Restore/Compare
- Artifacts：`QTreeView` 指向 artifacts 目录，支持右键下载或打开
- Console：后台日志流（`QPlainTextEdit`）并支持复制/保存

实现要点：每次运行生成 `artifacts/run_{timestamp}/` 目录并保存 `config.json`、模型文件、scaler、shap、run_info.json、run_log.txt 等产物；UI 支持双击将历史运行配置恢复到左侧表单

---

## 7. 弹窗 / 确认 / 高耗任务 UX 规范

高耗任务（如 Optuna、CMA-ES、PPO、全量 SHAP）必须弹出确认对话，内容建议包括：

- 粗略估算的计算量（例如 #folds × #boost_round × 模型复杂度）与磁盘需求提示（不承诺完成时间）
- 需要用户勾选“我理解并继续”的复选框才能确认执行

关于中止与错误处理：

- Stop 操作应向后台 worker 发送取消标志，worker 在安全点检查并优雅退出，同时保存中间产物；UI 显示“正在停止，正在写入产物”的提示
- 发生异常时在 Console 输出 traceback，并弹出简洁的 `QMessageBox` 显示错误摘要与产物路径，避免直接把完整 traceback 显示在弹窗中

---

## 8. 后端交互与并发模型（PySide6 实现要点）

推荐采用 `QThreadPool` + 自定义 `QRunnable`（或 `QThread` + worker QObject）模式，worker 通过信号向主线程报告状态：

- 推荐 signals：`log(str)`、`progress(float)`、`status(dict)`、`finished(result_dict)`、`error(str)`
- 所有长耗时任务（特征生成、交叉验证训练、Optuna、CMA-ES、PPO、SHAP 等）都在 worker 中执行，worker 负责把产物写入 `artifacts/run_xxx` 并在完成后发出 `finished` 信号
- 建议实现磁盘/内存缓存（例如使用 `joblib`）以避免重复计算，缓存 key 基于数据快照与配置哈希
- 进度上报应分阶段上报（比如 “Fold 3/10 — iter 120/500”），并映射到进度条文本，避免显示不可验证的 ETA

---

## 9. 配置 / 预设 / 导入导出 / 可复现性

- 使用 JSON schema 定义配置格式（顶层字段包括 data、feature_engineering、validation、model、mapping、training、artifacts），每次运行将 `config.json` 写入 artifacts 目录
- 内置预设（保守 / 平衡 / 激进），并允许用户保存自定义预设（建议存放 `~/.tas_presets/`）
- 每次运行保存复现元信息：`git_commit`、`python_env`（`pip freeze`）、`seed`、`data_hash`（文件 MD5 或样本哈希），记录到 `run_info.json`
- 可选集成 MLflow：后台 worker 将参数/指标/产物推送到 MLflow，并在 UI 中提供链接

---

## 10. 校验规则与自动诊断（重要约束）

在触发运行前执行轻量级校验并即时反馈：

- `embargo_days >= max_lag`（若不满足给出调整建议）
- 对于 LSTM：`seq_len <= len(train_df)`，否则警告并尝试使用训练历史回退方案
- 检查 `val_ratio` 与 `n_splits` 的冲突并提示
- 确认至少有 1 个特征被选中且与 `feature_cols` 一致
- 若选择使用 SHAP 且 `shap_sample > data_len`，自动 clamp 到数据长度
- 映射滑块应能反映历史分箱数与映射区间（若结果全为 0 则提示可能导致零换手）
- 对 Optuna/CMA-ES/PPO 等需检查是否设置随机种子以保证可复现性

此外，可以实现若干异步的自动诊断任务（低优先级），例如特征漂移检测、滚动信息系数稳定性评估等，超出阈值时触发告警

---

## 11. 视觉与交互细节（开发建议）

- 图表：优先使用 Plotly（交互性强）并通过 `QWebEngineView` 嵌入；若需导出高分辨率图片，可使用 Matplotlib 或 Plotly 的导出功能
- 主题：支持暗/亮主题切换，数值卡片可用 `QLabel` 或 `QLCDNumber` 风格美化
- 建议快捷键：Run = Ctrl+R、Stop = Ctrl+Shift+R、保存配置 = Ctrl+S、加载配置 = Ctrl+O
- 无障碍：所有控件应设置 `accessibleName` 与 `toolTip`
- 国际化（i18n）：建议将显示文本存入翻译文件以便中/英切换

---

## 12. 文件与产物组织约定

建议的 artifacts 目录结构（每次运行创建一个 `run_{timestamp}` 子目录）：

```
artifacts/
  run_2025-12-08_160102/
    config.json
    data_hash.txt
    model/
      lgbm_baseline.txt
      encoder.pth
      ppo_policy_latest.pth
    scalers/
      scaler.pkl
    shap/
      shap_train.npy
    plots/
      diagnostics.png
    run_info.json
    run_log.txt
    submission.csv
```

UI 可通过右侧 Artifacts 面板直接打开或下载上述文件。

---

## 附：JSON Schema 摘要（示例字段，便于前端渲染表单）

示例（简化）：

```json
{
  "data": {"train_csv": "train.csv", "test_csv": "test.csv", "missing_strategy": "median", "add_isnan": true},
  "validation": {"method":"purged_kfold","n_splits":10,"embargo_days":23,"label_horizon":1},
  "feature_engineering":{"lags":[1,2,5,10,21],"rolling_windows":[5,10,21,63],"ewma_span":21,"target_features":375,"use_shap":true,"shap_sample":500},
  "model":{"type":"lightgbm","params":{"learning_rate":0.02,"num_leaves":31,"min_child_samples":25}},
  "mapping":{"type":"cma_es","apply_isotonic":true,"cma_params":{"iters":150,"init":[1,1,0.5]}},
  "training":{"quick_run":true,"seed":42,"pretrain_epochs":5,"epochs":40},
  "artifacts":{"save_dir":"./artifacts"}
}
```


