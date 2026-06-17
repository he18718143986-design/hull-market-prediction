# 背景


我希望实现一个面向“标普500 日级回报/仓位分配”的股票预测软件，目标与 Kaggle 比赛  **Hull Tactical Market Prediction一样（https://www.kaggle.com/competitions/hull-tactical-market-prediction）** ：

* 输入：过去几十年（示例为 36 年）每个交易日与美股相关的市场/宏观/利率/情绪/动量等特征 + `forward_returns` + `risk_free_rate`。
* 输出：对每个交易日 `date_id=n` 的投资仓位（0 到 2），并根据次日真实回报计算策略表现。
* 评价：基于比赛的**volatility-adjusted Sharpe (adjusted_sharpe)** 指标（含波动罚项与回报差异罚项），具体代码在https://www.kaggle.com/code/metric/hull-competition-sharpe。

我希望把整个流程做成 **交互式 仪表盘** ，支持多模型（传统 ML / 深度学习 / 强化学习）切换、即时 Quick Run、训练过程可视化、SHAP 解释、实验记录与对比、以及“把最后 180 行当作真实竞赛数据”的模拟（逐日滚动重训练）。

# 目标

1. 上传 `train.csv`（具体介绍在https://www.kaggle.com/competitions/hull-tactical-market-prediction/data）进行实验。
2. 能在 UI 中选择/切换多种模型（多种机器学习算法：LightGBM等； 多种神经网络模型：MLP, LSTM等；多种强化学习算法：PPO等）。
3. 能做 Quick Run（挑选数据量、以及其他训练参数）查看改动对特征重要性、回测指标和仓位序列的即时影响。
4. 训练过程可视化（每 epoch/iter 的 loss、val adjusted_sharpe、RL update stats等，参考tensorboard），并可以中断/保存 checkpoint。
5. 支持 SHAP（TreeSHAP）和 permutation importance，以及单点 force plot。
6. 将 `train.csv` 最后 180 行当作“真实竞赛数据”进行逐日滚动模拟（含 retrain_every_days），以模拟实盘表现。
7. 每次实验结果自动保存（config + metrics + artifact 路径），可在 Experiments 页对比 top-k runs。

# 高层架构

推荐模块（文件/目录）与职责：

* `app.py`： PySide6 主界面，负责交互、参数收集、调用 trainers 与后端功能、展示图表与实验记录。
* `feature.py`：数据加载、缺失值处理、时间序列特征（mom/vol/zscore/lag）生成、特征选择、map_return_to_position 等。
* trainers/`lightgbm.py `：LightGBM（可保留作为 baseline）。
* `trainers/pytorch_trainer.py`：可导入的 PyTorch 训练器（LSTM/MLP），主函数 `train_torch_model(params, progress_callback=None, stop_event=None)`，并提供 `predict_full_series`、`evaluate_model_full_series_and_report`。
* `trainers/ppo_trainer.py`：PPO + `MarketEnv`，主函数 `ppo_train(params, progress_callback=None, stop_event=None)`，并写 `training_logs.csv`。
* `backtest.py`：回测逻辑与 scoring。
* `utils.py`：实验日志存取（`experiments.json`）、文件工具等。
* `requirements.txt` 与 `README_RUN.md`：依赖与运行说明。
* `runs/`：训练/模拟产出（checkpoints、plots、training_logs.csv、sim JSON 等）。

# 核心数据处理规则（防止数据泄露）

1. **时间序列严格按时间分割** ：所有训练/验证/测试分割必须是时间顺序（no random shuffle for temporal CV），避免未来信息泄露。
2. **Scaler/标准化只在训练集上 fit** ，然后用于验证/holdout（或在 expanding retrain 中，在扩展历史上重新 fit）。
3. **isnan 指示器** ：为每个数值特征生成 `_isnan` 列，作为模型输入，避免填充掩盖信息。
4. **LSTM 的序列填充** ：预测 holdout 前若序列不够长，用训练集尾部历史补齐（strictly historical only）。
5. **Holdout 模拟（last 180 rows）** ：把文件最后 H 行当作 “直播/比赛” 数据，仅在模拟中逐日或一次性暴露；训练集绝对不使用这些“未来”信息。

# 特征工程（建议步骤）

* 缺失值填充策略：`ffill`/`bfill`（时间序列）、或按列 median 填充。
* 构造基础特征：价格日回报、滚动 momentum（1,5,20 等）、滚动 volatility（5,20,60）、rolling zscore（window=252）并生成 lags。
* 标准化/缩放：对 numeric 特征使用 `StandardScaler` 在训练上 fit。
* 特征选择：按前缀（M,E,I,P,V,S,MOM,D）或 top-N、或后续用模型重要性/SHAP 自动筛选。
* 缓存：把处理好的特征写 parquet，用 `st.cache_data` 或本地缓存加速 Quick Run。

# 模型与训练策略

## 可用模型

* 传统 ML：Ridge（baseline），LightGBM（树模型，支持 fast SHAP）。
* 深度学习：Torch MLP（逐行回归到 position/proxy），Torch LSTM（序列到序列，输出时间步仓位或原始回报 proxy 然后映射）。
* 强化学习：PPO（Actor-Critic），需要自定义 `MarketEnv`。只作为高级模块，昂贵且需小心过拟合。

## 损失/目标

* 对神经网络，使用可微分的  **adjusted_sharpe_loss_double** ，直接把 adjusted-sharpe 风格目标作为损失进行微调（并加上 turnover penalty 与 L2 正则）。
* 也支持两阶段训练：先 supervised pretrain（预测 forward_returns 或 proxy），然后用 adjusted_sharpe loss 微调（可以稳定训练）。

## 训练细节

* **训练回调** ：在每 epoch/end 和每个重要事件调用 `progress_callback(dict)`，传回 `epoch`, `train_loss`, `val_adj_sharpe`, `best_val`, `checkpoint` 等。
* **Quick Run** ：交互模式下用较少数据/少 epoch/较小 batch，默认 epochs=3 / n_train 小片段，从而秒到分钟级反馈。Full Train 使用完整设定。
* **保存点** ：每当验证指标改善就保存 checkpoint（包含 `model_state`, `scaler`, `feature_cols`, `model_type`）。
* **中断** ：训练线程接收 `stop_event`（`threading.Event`），在合理点退出并保存状态。

# 训练可视化

* **后台线程** ：将训练放后台线程（`threading.Thread`）并通过 `progress_callback` 将指标推回主线程
* 绘图：`st.line_chart` 绘制 `train_loss`、`val_adj_sharpe`；RL 绘制 `avg_ret`、`avg_pos`、`avg_turnover`。
* SHAP：对 LightGBM 使用 `shap.TreeExplainer` 快速绘制 summary plot；对 NN 使用小样本 Kernel SHAP（或 permutation importance）。
* 单行情 force plot：可把 `shap.force_plot` 转为 HTML 并用 `st.components.v1.html` 嵌入。

# 回测与交易成本

* 回测逻辑 `compute_strategy_returns_with_costs`：

  * `strategy_returns = rf*(1-pos) + pos * forward_returns`（原始）
  * 交易成本包含：`per_trade_cost * turnover`、`spread_cost * (position changed)`、`turnover_penalty_coef * turnover`。
* 在 adjusted_sharpe 计算中加入策略收益（含成本），再计算 annualized Sharpe、波动罚项、return gap penalty。
* `turnover` = `abs(pos_t - pos_{t-1})`；高 turnover 会被 `turnover_penalty_coef` 惩罚并体现在损失/回测里。
* 在 UI 提供成本参数开关以模拟不同市场摩擦场景。

# Holdout / 模拟“最后180行”的两种模式（默认 H=180）

**滚动/实时模拟 (rolling)** ：

* 把最后 180 行看成逐日到来的数据；在每个到来日 t：用历史（train + 已暴露的 holdout[0..t-1]）训练或微调（或按固定周期重训练），预测当天仓位，保存收益，并把当天加入历史。
* 参数 `retrain_every_days` 决定重训练频率（0 = none, 1 = daily，建议 7 或 30 作为折中）。
* LSTM 在开头序列不足时用训练集尾部补齐（`train_history_df`）。
* 结果以时间序列形式保存并计算最终 adjusted_sharpe 与其他绩效指标。

# 强化学习（PPO）注意事项

* PPO算法在集成时推荐：

  * 把预训练与 PPO 训练分离，预训练可在 CPU 上完成，PPO 在 GPU/更大资源上运行。
  * 在 UI 中提供 Quick Preset（`n_updates`、`steps_per_update` 缩小）以便交互测试。
  * 实际上，在线滚动模拟对 RL 的实现复杂度高、资源消耗大；为了评估 POOL：训练好 policy 后在 holdout 上以确定性 mean action 做静态评估最稳妥。

# 实验管理 & 对比

* 一次 run 应自动存储：timestamp, model_type, hyperparams, features_count, metrics (AdjustedSharpe, Sharpe, CAGR, vol, maxDD), artifact paths (checkpoint, logs, plots)。存入 `experiments.json` 或使用 MLflow/W&B（生产推荐）。
* Experiments 页面提供 Top-k 按 adjusted_sharpe 排序、并支持 multi-select 比较 cumulative return 曲线与 KPI。
* Quick Run 与 Full Train 区分并记录（便于复现）。

# UI（Dashboard）设计建议（专业仪表盘）

* 页面布局（wide, multi-column）：

  * 顶部：Run Controls（模型、Quick/Full、seed、训练/停止/导出）
  * 左栏：数据与特征工程面板（上传/预处理/特征选择/运行 pipeline）
  * 中央：训练可视化（实时曲线）、SHAP summary、特征重要性（并列）
  * 右侧：回测结果（KPIs 矩阵：Adjusted Sharpe, CAGR, Volatility, MaxDD, Turnover）、累积收益比较图（策略 vs 市场）
  * 底部或新 tab：Experiments 列表、多 run 对比、下载 artifacts
* 交互：更改特征或超参 → 点击 Quick Run → 秒级返回图表 & KPI；若想 Full Train 则后台长训练并在完成后报告。
* SHAP/解释视图：全局 summary + 单日 force plot + 条形 top features；对树模型直接显示 TreeSHAP，NN 用近似方法或小样本 Kernel SHAP。

# 工程实现说明（你已要求的完整重构）

* 已实现并推荐结构：`trainers/pytorch_trainer.py` 与 `trainers/ppo_trainer.py`
* 关键接口：
  * `train_torch_model(params, progress_callback=None, stop_event=None)` → 返回 meta（best_checkpoint, best_val, run_dir）并通过 `progress_callback(dict)` 通知进度。
  * `ppo_train(params, progress_callback=None, stop_event=None)` → 每次 update 写 `training_logs.csv` 并通过 callback 汇报 stats。
  * `simulate_live_holdout(train_csv, holdout_size=180, model_params=..., mode='rolling', retrain_every_days=..., progress_callback=...)` → 执行 holdout 模拟并返回时间序列结果。
  * `predict_full_series(model, df, feature_cols, scaler, device, seq_len, model_type)` → 用于静态/滚动预测。
* 日志/Artifact 存放：每 run 保存到 `runs/<timestamp>/`，包含 checkpoint、plots、training_logs.csv、simulate JSON 等。

# 性能与硬件建议

* 小规模 Quick Run（少样本、少 epoch）：可在 CPU 完成交互测试。
* Full Train（LSTM/large NN / PPO）：强烈建议 GPU（CUDA）支持；PPO 通常需要更多时间与更大内存。
* 避免在 UI 主进程堵塞训练：使用线程或更稳健的后台作业队列（Redis+RQ/Celery）用于生产部署。

# 可测性/调试/常见问题

* 常见问题：gradient explosion（clip grad）、zero std 导致 sharpe 计算异常（捕获并早停）、SHAP 计算慢（采样/限制 sample size）。
* 调试建议：先用 Quick Run 复现小样本训练；在 Full Train 前确认 `train_torch_model` 在本地能稳定保存 checkpoint 并用 `predict_full_series` 进行正确预测。
* 记录随机种子、数据版本、特征列表以保证可重复性。
* 用 `val_ratio` 监控验证集表现，避免过拟合。

# 安全 & 合规

* 不要在 UI 中泄露任何用户隐私数据（如果数据含有敏感信息，移除或脱敏）。
* Model artifacts 若有权限要求，限制下载权限。
* 若用于真实资金，注意合规与风险披露（模型仅作研究参考）。
