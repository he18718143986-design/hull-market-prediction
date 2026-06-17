# Hull Market Prediction / 市场仓位预测与回测平台

> Interactive desktop dashboard for S&P 500 daily allocation prediction, aligned with
> the [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
> Kaggle competition. Supports LightGBM / PyTorch (MLP, LSTM) / PPO, volatility-adjusted
> Sharpe optimization, rolling holdout simulation, and experiment comparison.
>
> 面向标普 500 日级仓位分配的交互式桌面仪表盘，对标 Kaggle Hull Tactical 竞赛。
> 支持 LightGBM / PyTorch（MLP、LSTM）/ PPO，可微 adjusted-Sharpe 损失、
> 180 日滚动模拟与实验对比。

---

## Highlights / 项目亮点

- **Multi-model dashboard / 多模型仪表盘**：PySide6 GUI with Quick Run, full training,
  real-time loss/Sharpe curves, SHAP feature importance, and experiment history.
- **Adjusted Sharpe as loss / 可微竞赛指标**：`train_adjusted_sharpe.py` implements
  differentiable volatility-adjusted Sharpe (with turnover penalty) for direct NN fine-tuning.
- **Anti-leakage design / 防泄露设计**：strict temporal splits, scaler fit on train only,
  `_isnan` indicators, LSTM history padding from past data only.
- **Rolling holdout simulation / 滚动模拟**：treat last 180 rows as live competition data,
  retrain every N days, one-step-ahead position prediction.
- **End-to-end RL pipeline / 端到端 RL**：`supervised_pretrain.py` → encoder →
  `ppo_market_agent.py` / `train_full_pipeline.py` for PPO fine-tuning.

## Architecture / 模块结构

| File | Role |
|---|---|
| `hull/app.py` | **Main GUI entry** — PySide6 dashboard (model switch, training viz, experiments). 主界面入口。 |
| `hull/simulator.py` | Rolling holdout simulation workers (daily retrain + predict). 滚动模拟。 |
| `hull/trainer.py`, `lgb_runner.py` | Training job dispatch for NN / LightGBM. 训练调度。 |
| `hull/job_queue.py`, `worker.py` | Background job queue for long-running trains. 后台任务队列。 |
| `baseline.py` | LightGBM baseline with feature engineering & CV. LightGBM 基线。 |
| `train_adjusted_sharpe.py` | PyTorch MLP/LSTM trainer with adjusted-Sharpe loss. 核心训练脚本。 |
| `train_full_pipeline.py` | Supervised pretrain → PPO fine-tune end-to-end. 预训练 + PPO 全流程。 |
| `ppo_market_agent.py` | PPO market environment & policy. PPO 智能体。 |
| `score.py` | Official Kaggle volatility-adjusted Sharpe metric. 竞赛评分函数。 |
| `loss.py` | Differentiable loss variants. 可微损失。 |

```
train.csv (user-provided, from Kaggle)
    ↓  feature engineering + temporal split
baseline.py / train_adjusted_sharpe.py / train_full_pipeline.py
    ↓  position ∈ [0, 2]
score.py  —  adjusted Sharpe backtest
    ↓
hull/app.py  —  GUI visualization, SHAP, rolling simulation, experiment log
```

## Quick start / 快速开始

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download competition data

Place `train.csv` from
[Kaggle Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data)
in the project root (or specify path in the GUI).

### 3. Launch the dashboard

```bash
python hull/app.py
```

### 4. CLI training (no GUI)

```bash
# LightGBM baseline
python baseline.py --train_csv train.csv

# PyTorch MLP/LSTM with adjusted-Sharpe loss
python train_adjusted_sharpe.py --train_csv train.csv --model lstm --visualize

# Supervised pretrain → PPO fine-tune
python train_full_pipeline.py
```

## Not included / 未包含项

| Excluded | Reason |
|---|---|
| `train.csv` / `test.csv` (~11 MB) | Kaggle competition data — download separately |
| `checkpoints/`, `*.pth`, `gui_models/` | Trained model weights |
| `test/`, `预测/`, `股票/` | Experimental notebooks & iteration code |
| `Hull Tactical - Market Prediction/` | Earlier prototype copy |
| `gui_app.py` | Superseded by `hull/app.py` |
| Root `c.py` / `d.py` / `e.py` | Scratch experiment scripts |

See `docs/README.md` (product spec) and `docs/README_UI.md` (UI layout) for full design docs.

## Tech stack / 技术栈

`Python 3.8+` · `PySide6` · `PyTorch` · `LightGBM` · `stable-baselines3` (PPO) · `plotly` · `pyqtgraph`

## License

[MIT](LICENSE)
