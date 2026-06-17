#!/usr/bin/env python3
"""
train_adjusted_sharpe_merged.py

合并版训练脚本：full-series 验证 + double-precision 可微损失 + 可视化 + 数据缺失指示器

Usage:
    python train_adjusted_sharpe_merged.py --train_csv train.csv [--test_csv test.csv] [--model {lstm,mlp}] [options]

主要参数（示例）:
    --train_csv train.csv
    --test_csv test.csv
    --model lstm
    --seq_len 252
    --visualize
"""
import os
import argparse
import math
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tempfile
import time
import json

import matplotlib
# Use a non-interactive backend to avoid opening GUI windows when plotting
# from background threads (prevents crashes when training runs off the main thread).
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib import font_manager

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import shutil
import pickle


# ---------------------------
# 小工具：设置中文字体（可选）
# ---------------------------
def setup_chinese_font():
    mac_fonts = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'Hiragino Sans GB', 'Heiti SC']
    win_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun']
    linux_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    font_candidates = mac_fonts + win_fonts + linux_fonts
    found_font = None
    for font_name in font_candidates:
        if font_name in available_fonts:
            found_font = font_name
            break
    if found_font:
        cur = matplotlib.rcParams.get('font.sans-serif', [])
        if isinstance(cur, str):
            cur = [cur]
        matplotlib.rcParams['font.sans-serif'] = [found_font] + [f for f in cur if f != found_font]
    matplotlib.rcParams['axes.unicode_minus'] = False

# call optionally
setup_chinese_font()

# ---------------------------
# 常量 & 随机种子
# ---------------------------
TRADING_DAYS = 252.0
EPS = 1e-12

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# 数据处理类（两种模式）
# ---------------------------
class TimeSeriesDataset(Dataset):
    """
    为LSTM创建重叠的序列长度为seq_len的序列
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], seq_len: int = 252, step: int = 1, scaler: Optional[StandardScaler]=None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.step = step
        self.scaler = scaler

        self.df = df.reset_index(drop=True).copy()
        X = self.df[self.feature_cols].ffill().fillna(0.0).values.astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        self.X = X
        self.forward_returns = self.df['forward_returns'].values.astype(np.float32)
        self.risk_free = self.df['risk_free_rate'].values.astype(np.float32)
        self.indices = []
        N = len(self.df)
        for start in range(0, N - seq_len + 1, step):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.seq_len
        return {
            'features': torch.from_numpy(self.X[s:e]),          # (T, D)
            'forward_returns': torch.from_numpy(self.forward_returns[s:e]),
            'risk_free': torch.from_numpy(self.risk_free[s:e]),
        }

class FinancialDataset(Dataset):
    """
    为MLP创建每个行是一个日期的数据集（如果提供了滞后特征，则使用滞后特征）
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        df = df.replace([np.inf, -np.inf], np.nan).copy()
        # 一次性创建所有 isnan 列，避免 DataFrame 碎片化
        isnan_cols = {}
        for c in feature_cols:
            isnan_name = f"{c}_isnan"
            # only create indicator if not already present
            if isnan_name not in df.columns:
                isnan_cols[isnan_name] = df[c].isna().astype(np.float32)
        if isnan_cols:
            df = pd.concat([df, pd.DataFrame(isnan_cols, index=df.index)], axis=1)
        # 填充中位数
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        self.features = df[feature_cols].values.astype(np.float32)
        self.forward_returns = df['forward_returns'].values.astype(np.float32)
        self.risk_free = df['risk_free_rate'].values.astype(np.float32)
        self.date_id = df['date_id'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.forward_returns[idx]), torch.tensor(self.risk_free[idx])

# ---------------------------
# 模型：LSTM 与 MLP 两种
# ---------------------------
class SimpleLSTMPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=1, dropout=0.1):
        super().__init__()
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=lstm_dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_dim//2), 1)
        )
    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        raw = self.head(out)  # (B, T, 1)
        return raw.squeeze(-1)  # (B, T)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        raw = self.net(x)
        pos = 2.0 * torch.sigmoid(raw)
        return pos.squeeze(-1)

# ---------------------------
# 双精度可微损失（尽量与竞赛 score() 保持数值一致）
# ---------------------------
def adjusted_sharpe_loss_double(raw_pos, forward_returns, risk_free, lambda_turnover=1.0, l2_reg=0.0, model=None):
    """
    数值稳定且向量化的调整后夏普损失（基于 loss.AdjustedSharpeLoss 实现），
    接受 raw_pos（logits）并返回标量损失与 stats 字典。
    保持与旧接口兼容：输入可以是 (B, T) 或 (B,)（MLP 情况）。
    """
    if raw_pos.dim() == 1:
        raw_pos = raw_pos.unsqueeze(1)
        forward_returns = forward_returns.unsqueeze(1)
        risk_free = risk_free.unsqueeze(1)

    # positions in [0,2]
    pos = 2.0 * torch.sigmoid(raw_pos)  # (B, T)

    strategy_returns = risk_free * (1.0 - pos) + pos * forward_returns
    strategy_excess = strategy_returns - risk_free
    market_excess = forward_returns - risk_free

    # 使用 log1p/mean/exp 计算几何平均以提高数值稳定性
    # clamp 至大于 -1
    min_clip = -1.0 + 1e-12
    se = torch.clamp(strategy_excess, min=min_clip)
    me = torch.clamp(market_excess, min=min_clip)

    # mean of log1p across time axis
    strategy_geo_mean = torch.exp(torch.mean(torch.log1p(se), dim=1)) - 1.0
    market_geo_mean = torch.exp(torch.mean(torch.log1p(me), dim=1)) - 1.0

    # 标准差（保持与 numpy/pandas ddof=1 的行为，当序列长度为1时改为无偏=False 避免 nan）
    n = raw_pos.shape[1]
    use_unbiased = True if n > 1 else False
    strategy_std = torch.std(strategy_returns.double(), dim=1, unbiased=use_unbiased).to(raw_pos.dtype) + EPS
    market_std = torch.std(forward_returns.double(), dim=1, unbiased=use_unbiased).to(raw_pos.dtype) + EPS

    sharpe = (strategy_geo_mean / strategy_std) * math.sqrt(TRADING_DAYS)
    strategy_vol = strategy_std * math.sqrt(TRADING_DAYS) * 100.0
    market_vol = market_std * math.sqrt(TRADING_DAYS) * 100.0

    excess_vol = torch.relu(strategy_vol / (market_vol + 1e-12) - 1.2)
    vol_penalty = 1.0 + excess_vol

    return_gap = torch.relu((market_geo_mean - strategy_geo_mean)) * 100.0 * TRADING_DAYS
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty + 1e-12)

    turnover = torch.mean(torch.abs(pos[:, 1:] - pos[:, :-1]), dim=1) if pos.shape[1] > 1 else torch.zeros(pos.shape[0], device=pos.device)

    loss_val = - adjusted_sharpe.mean() + lambda_turnover * turnover.mean()

    if l2_reg > 0.0 and model is not None:
        l2 = torch.tensor(0.0, device=raw_pos.device)
        for p in model.parameters():
            l2 = l2 + torch.sum(p ** 2)
        loss_val = loss_val + l2_reg * l2

    stats = {
        'adjusted_sharpe_per_seq': adjusted_sharpe.detach().cpu().numpy(),
        'turnover_per_seq': turnover.detach().cpu().numpy(),
        'strategy_vol_per_seq': strategy_vol.detach().cpu().numpy(),
        'market_vol_per_seq': market_vol.detach().cpu().numpy(),
    }
    return loss_val, stats


def adjusted_sharpe_loss_via_losspy(raw_pos, forward_returns, risk_free, lambda_turnover=1.0, l2_reg=0.0, model=None):
    """
    Wrapper that uses `loss.AdjustedSharpeLoss` implementation (from loss.py) to compute
    per-sequence adjusted sharpe, then composes the final loss with turnover and L2
    to match behavior of `adjusted_sharpe_loss_double`.
    """
    # ensure shapes similar to adjusted_sharpe_loss_double
    if raw_pos.dim() == 1:
        raw_pos = raw_pos.unsqueeze(1)
        forward_returns = forward_returns.unsqueeze(1)
        risk_free = risk_free.unsqueeze(1)

    # Directly use the local adjusted_sharpe_loss_double implementation (no external module call)
    loss_val, stats = adjusted_sharpe_loss_double(raw_pos, forward_returns, risk_free, lambda_turnover=lambda_turnover, l2_reg=l2_reg, model=model)
    # adjusted_sharpe_loss_double returns adjusted_sharpe_per_seq as positive sharpe values; wrapper historically returned positives
    # keep turnover_per_seq present
    return loss_val, {'adjusted_sharpe_per_seq': stats.get('adjusted_sharpe_per_seq'), 'turnover_per_seq': stats.get('turnover_per_seq')}

# ---------------------------
# 完整时间序列预测 & numpy评分（与竞赛一致）
# ---------------------------
def compute_adjusted_sharpe_numpy(positions, forward_returns, risk_free_rates):
    """
    输入：positions, forward_returns, risk_free_rates: 1D numpy数组对齐
    输出：adjusted_sharpe: 调整后的夏普比率
    stats: 统计字典
    """
    if len(positions) < 2:
        return float('nan'), {}

    strategy_returns = risk_free_rates * (1.0 - positions) + positions * forward_returns
    strategy_excess = strategy_returns - risk_free_rates
    market_excess = forward_returns - risk_free_rates

    se = np.clip(strategy_excess, a_min=-1.0 + 1e-12, a_max=None)
    me = np.clip(market_excess, a_min=-1.0 + 1e-12, a_max=None)

    strategy_geo_mean = np.exp(np.mean(np.log1p(se))) - 1.0
    market_geo_mean = np.exp(np.mean(np.log1p(me))) - 1.0

    strategy_std = np.std(strategy_returns, ddof=1)  # pandas default ddof=1
    market_std = np.std(forward_returns, ddof=1)

    if strategy_std == 0 or market_std == 0:
        raise ValueError("Zero std encountered in compute_adjusted_sharpe_numpy")

    sharpe = (strategy_geo_mean / strategy_std) * math.sqrt(TRADING_DAYS)
    strategy_vol = strategy_std * math.sqrt(TRADING_DAYS) * 100.0
    market_vol = market_std * math.sqrt(TRADING_DAYS) * 100.0

    excess_vol = max(0.0, strategy_vol / (market_vol + 1e-12) - 1.2)
    vol_penalty = 1.0 + excess_vol

    return_gap = max(0.0, (market_geo_mean - strategy_geo_mean) * 100.0 * TRADING_DAYS)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty + 1e-12)

    stats = {
        'adjusted_sharpe': float(adjusted_sharpe),
        'sharpe': float(sharpe),
        'strategy_vol': float(strategy_vol),
        'market_vol': float(market_vol),
        'turnover_mean': float(np.mean(np.abs(np.diff(positions)))) if len(positions) > 1 else 0.0,
        'cumulative_strategy_excess': float(np.prod(1 + strategy_excess) - 1.0),
        'cumulative_market_excess': float(np.prod(1 + market_excess) - 1.0),
    }
    return float(adjusted_sharpe), stats


# ============================================================================
# 工具函数：评分函数
# ============================================================================

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = None) -> float:
    """计算调整后夏普比率"""
    # Accept either 'prediction' or 'allocation'
    if 'prediction' in submission.columns:
        preds = submission['prediction']
    elif 'allocation' in submission.columns:
        preds = submission['allocation']
    else:
        raise ParticipantVisibleError('Submission must contain column "prediction" or "allocation"')

    # Ensure numeric and replace NaN with 0 (conservative)
    preds = pd.to_numeric(preds, errors='coerce').fillna(0.0).astype(float)
    if not np.issubdtype(preds.dtype, np.floating):
        raise ParticipantVisibleError('Predictions must be numeric')

    if preds.max() > 2 + 1e-12:
        raise ParticipantVisibleError(f'Position of {preds.max()} exceeds maximum of 2')
    if preds.min() < -1e-12:
        raise ParticipantVisibleError(f'Position of {preds.min()} below minimum of 0')

    sol = solution.copy().reset_index(drop=True)
    sol['position'] = preds.reset_index(drop=True)

    # strategy returns: rf * (1 - pos) + pos * forward_returns
    if 'forward_returns' not in sol.columns or 'risk_free_rate' not in sol.columns:
        raise ParticipantVisibleError('Solution must contain "forward_returns" and "risk_free_rate" columns')
    sol['strategy_returns'] = sol['risk_free_rate'] * (1 - sol['position']) + sol['position'] * sol['forward_returns']

    # strategy excess returns and market excess returns
    strategy_excess = sol['strategy_returns'] - sol['risk_free_rate']
    market_excess = sol['forward_returns'] - sol['risk_free_rate']

    n = len(sol)
    if n < 2:
        raise ParticipantVisibleError('Not enough samples to compute score')

    # Cumulative geometric returns then annualize (match competition)
    strategy_excess_cumulative = (1 + strategy_excess).prod()
    strategy_mean_excess = strategy_excess_cumulative ** (1.0 / n) - 1.0

    strategy_std = float(sol['strategy_returns'].std(ddof=1))
    if strategy_std == 0:
        raise ParticipantVisibleError('Division by zero, strategy std is zero')

    # annualize by trading days
    trading_days_per_yr = 252
    sharpe = (strategy_mean_excess / strategy_std) * np.sqrt(trading_days_per_yr)
    strategy_volatility = float(strategy_std * np.sqrt(trading_days_per_yr) * 100.0)

    # market
    market_excess_cumulative = (1 + market_excess).prod()
    market_mean_excess = market_excess_cumulative ** (1.0 / n) - 1.0

    market_std = float(sol['forward_returns'].std(ddof=1))
    if market_std == 0:
        raise ParticipantVisibleError('Division by zero, market std is zero')
    market_volatility = float(market_std * np.sqrt(trading_days_per_yr) * 100.0)

    # volatility penalty
    excess_vol = max(0.0, strategy_volatility / market_volatility - 1.2) if market_volatility > 0 else 0.0
    vol_penalty = 1.0 + excess_vol

    # return penalty (annualized gap squared / 100)
    return_gap = max(0.0, (market_mean_excess - strategy_mean_excess) * 100.0 * trading_days_per_yr)
    return_penalty = 1.0 + (return_gap ** 2) / 100.0

    adjusted_sharpe = sharpe / (vol_penalty * return_penalty)
    return min(float(adjusted_sharpe), 1_000_000.0)


def score_with_fallback(date_ids, positions, fr, rf):
    """
    输入：date_ids, positions, fr, rf: 1D numpy数组对齐
    输出：adjusted_sharpe: 调整后的夏普比率
    stats: 统计字典
    """
    sol = pd.DataFrame({
        'date_id': date_ids,
        'forward_returns': fr,
        'risk_free_rate': rf
    })
    sub = pd.DataFrame({
        'date_id': date_ids,
        'prediction': positions
    })
    val = score(sol.copy(), sub.copy(), row_id_column_name='date_id')
    adj, stats = compute_adjusted_sharpe_numpy(positions, fr, rf)
    stats['score_used_competition'] = True
    return float(adj), stats

# ---------------------------
# 完整时间序列预测（处理两种模型）
# ---------------------------
def predict_full_series(model, df, feature_cols, scaler, device, seq_len, model_type='lstm', train_history_df: Optional[pd.DataFrame]=None):
    model.eval()
    df_sorted = df.sort_values('date_id').reset_index(drop=True).copy()

    # 分离原始特征和 isnan 特征
    base_feature_cols = [c for c in feature_cols if not c.endswith('_isnan')]
    isnan_feature_cols = [c for c in feature_cols if c.endswith('_isnan')]
    
    # 处理预处理：添加isnan指示器和填充中位数（与训练相同）
    # 只为原始特征创建 isnan 列（如果还没有）
    isnan_cols = {}
    for c in base_feature_cols:
        isnan_col_name = f"{c}_isnan"
        if isnan_col_name in feature_cols and isnan_col_name not in df_sorted.columns:
            isnan_cols[isnan_col_name] = df_sorted[c].isna().astype(np.float32)
    if isnan_cols:
        df_sorted = pd.concat([df_sorted, pd.DataFrame(isnan_cols, index=df_sorted.index)], axis=1)
    
    # 填充缺失值
    df_sorted[base_feature_cols] = df_sorted[base_feature_cols].fillna(df_sorted[base_feature_cols].median())
    
    # 确保所有 feature_cols 中的列都存在，按 feature_cols 的顺序提取
    X = df_sorted[feature_cols].values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)

    N = len(df_sorted)
    positions = []
    fr_list = []
    rf_list = []
    date_ids = []

    device = next(model.parameters()).device

    if model_type == 'lstm':
        # 需要滑动窗口；如果df短于seq_len，使用train_history_df尾
        needs_history = (N < seq_len)
        if needs_history:
            if train_history_df is None:
                raise ValueError("Need train_history_df when df shorter than seq_len for LSTM predict")
            # 确保 train_history_df 有所有需要的特征列
            hist_df = train_history_df.copy()
            missing_cols = [c for c in feature_cols if c not in hist_df.columns]
            for c in missing_cols:
                if c.endswith('_isnan'):
                    base_name = c.replace('_isnan', '')
                    if base_name in hist_df.columns:
                        hist_df[c] = hist_df[base_name].isna().astype(np.float32)
            hist = hist_df[feature_cols].ffill().fillna(0.0).values.astype(np.float32)
            if scaler is not None:
                hist = scaler.transform(hist)
            history = hist[-seq_len:]
        else:
            history = None

        start_idx = seq_len if not needs_history else 0
        with torch.no_grad():
            for idx in range(start_idx, N):
                if not needs_history:
                    window = X[idx - seq_len: idx]
                else:
                    if idx == 0:
                        window = history
                    else:
                        needed_from_test = idx
                        needed_from_history = seq_len - needed_from_test
                        if needed_from_history > 0:
                            window = np.vstack([history[-needed_from_history:], X[:needed_from_test]])
                        else:
                            window = X[idx - seq_len: idx]
                w = torch.from_numpy(window).unsqueeze(0).to(device)
                raw = model(w)  # (1, seq_len)
                raw_last = raw[0, -1]
                pos = float(2.0 * torch.sigmoid(raw_last).cpu().item())
                pos = float(np.clip(pos, 0.0, 2.0))
                row = df_sorted.iloc[idx]
                positions.append(pos)
                fr_list.append(float(row['forward_returns']))
                rf_list.append(float(row['risk_free_rate']))
                date_ids.append(row['date_id'])
    else:
        # MLP: 逐行预测（如果目标中有NaN，则跳过第一行）
        with torch.no_grad():
            batch_size = 512
            for i in range(0, N, batch_size):
                x_batch = torch.from_numpy(X[i:i+batch_size]).to(device)
                pos_batch = model(x_batch)
                pos_batch = pos_batch.clamp(0.0, 2.0).cpu().numpy()
                positions.extend(pos_batch.tolist())
                fr_list.extend(df_sorted['forward_returns'].values[i:i+batch_size].tolist())
                rf_list.extend(df_sorted['risk_free_rate'].values[i:i+batch_size].tolist())
                date_ids.extend(df_sorted['date_id'].values[i:i+batch_size].tolist())

    return np.array(date_ids), np.array(positions), np.array(fr_list), np.array(rf_list)

# ---------------------------
# 可视化辅助函数
# ---------------------------
def plot_positions_and_returns(date_ids, positions, forward_returns, risk_free, title=None, savepath=None):
    """
    绘制仓位和回报的图表
    输入：date_ids, positions, forward_returns, risk_free: 1D numpy数组对齐
    输出：None
    """
    if len(positions) == 0:
        return
    # date_ids 是数值 ID，不是日期时间，直接使用作为 x 轴
    dates = date_ids
    strategy_daily = (risk_free * (1 - positions) + positions * forward_returns)
    strategy_excess = strategy_daily - risk_free
    market_excess = forward_returns - risk_free

    cum_strategy = np.cumprod(1 + strategy_excess) - 1
    cum_market = np.cumprod(1 + market_excess) - 1

    window = 63
    rolling_sharpe = pd.Series(strategy_excess).rolling(window).mean() / (pd.Series(strategy_excess).rolling(window).std() + 1e-8) * math.sqrt(TRADING_DAYS)

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(dates, positions, label='position')
    ax[0].set_ylabel('position (0-2)')
    ax[0].legend()
    ax[1].plot(dates, cum_strategy, label='strategy cum excess')
    ax[1].plot(dates, cum_market, label='market cum excess', linestyle='--')
    ax[1].set_ylabel('cumulative excess return')
    ax[1].legend()
    ax[2].plot(dates, rolling_sharpe, label=f'rolling {window}d sharpe')
    ax[2].set_ylabel('rolling sharpe')
    ax[2].set_xlabel('Date ID')
    ax[2].legend()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def plot_turnover_hist(positions, savepath=None):
    """
    绘制每日换手的直方图
    输入：positions: 1D numpy数组
    输出：None
    """
    if len(positions) <= 1:
        return
    turnover = np.abs(np.diff(positions))
    fig, ax = plt.subplots(1,1, figsize=(10,3))
    ax.plot(turnover)
    ax.set_title('Per-day turnover')
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def feature_drift_report(train_df, other_df, feature_cols, topk=30):
    """
    生成特征漂移报告
    输入：train_df, other_df, feature_cols: 训练数据集，其他数据集，特征列
    输出：DataFrame，包含特征，训练-其他均值差异，训练/其他标准差比率
    """
    rows = []
    for c in feature_cols:
        a = train_df[c].dropna()
        b = other_df[c].dropna()
        if len(a)==0 or len(b)==0:
            continue
        rows.append((c, float(a.mean()-b.mean()), float((a.std()+1e-9)/(b.std()+1e-9))))
    rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)[:topk]
    return pd.DataFrame(rows, columns=['feature','mean_diff(train-other)','std_ratio(train/other)'])

# ---------------------------
# 完整时间序列验证的训练循环
# ---------------------------
def train(args, progress_hook=None):
    """
    训练模型
    输入：args: 参数
    输出：None
    """
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("Device:", device)
    if progress_hook:
        try:
            progress_hook(f"Device: {device}")
        except Exception:
            pass

    df = pd.read_csv(args.train_csv)
    df = df.sort_values('date_id').reset_index(drop=True)

    # 删除不是特征的列
    drop_cols = {'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'}
    base_feature_cols = [c for c in df.columns if c not in drop_cols]

    # 构建滞后列（如果使用MLP，则模型可以看到昨天的信息）
    if args.model == 'mlp':
        for col in ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']:
            df[f'lag1_{col}'] = df[col].shift(1)
        # 删除第一行
        df = df.iloc[1:].reset_index(drop=True)
        # 重新计算base_feature_cols以包含滞后列
        base_feature_cols = [c for c in df.columns if c not in drop_cols]

    # 创建 isnan 指示器（可选）并根据缺失策略填充特征，确保训练数据无 NaN
    strategy = getattr(args, 'missing_value_strategy', 'median')
    try:
        if strategy is None:
            strategy = 'median'
        strategy = str(strategy).lower()
    except Exception:
        strategy = 'median'

    # 在创建 isnan 指示器前，强制把 base 特征和目标列转换为数值型（把非数值转为 NaN），
    # 以避免字符串/空串列在后续数值处理链中引起问题。
    try:
        df[base_feature_cols] = df[base_feature_cols].apply(lambda s: pd.to_numeric(s, errors='coerce'))
    except Exception:
        pass
    # 同样确保目标列为数值
    try:
        if 'forward_returns' in df.columns:
            df['forward_returns'] = pd.to_numeric(df['forward_returns'], errors='coerce')
        if 'risk_free_rate' in df.columns:
            df['risk_free_rate'] = pd.to_numeric(df['risk_free_rate'], errors='coerce')
        if 'market_forward_excess_returns' in df.columns:
            df['market_forward_excess_returns'] = pd.to_numeric(df['market_forward_excess_returns'], errors='coerce')
    except Exception:
        pass

    # 一次性创建所有 isnan 列，避免 DataFrame 碎片化
    isnan_cols = {}
    for c in base_feature_cols:
        isnan_cols[f"{c}_isnan"] = df[c].isna().astype(np.float32)
    if isnan_cols:
        df = pd.concat([df, pd.DataFrame(isnan_cols, index=df.index)], axis=1)

    # 应用缺失处理策略（默认：中位数填充）
    if strategy == 'ffill':
        df[base_feature_cols] = df[base_feature_cols].ffill().bfill().fillna(0.0)
    elif strategy == 'zero' or strategy == '0' or strategy == 'none':
        # treat 'none' conservatively as zero-fill to avoid NaNs during training
        df[base_feature_cols] = df[base_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif strategy == 'interpolate' or strategy == 'interp':
        df[base_feature_cols] = df[base_feature_cols].interpolate().fillna(df[base_feature_cols].median())
    else:
        # default: median
        try:
            df[base_feature_cols] = df[base_feature_cols].fillna(df[base_feature_cols].median())
        except Exception:
            df[base_feature_cols] = df[base_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 按时间分割
    n = len(df)
    val_n = int(n * args.val_ratio)
    train_df = df.iloc[: n - val_n].reset_index(drop=True)
    val_df = df.iloc[n - val_n:].reset_index(drop=True)

    # 最终特征列
    feature_cols = [c for c in train_df.columns if c not in drop_cols and c != 'date_id']
    print(f"Feature count: {len(feature_cols)}; example: {feature_cols[:8]}")

    # 在训练特征上拟合scaler
    scaler = StandardScaler()
    train_X = train_df[feature_cols].values.astype(np.float32)
    # sanitize training matrix: replace NaN/inf with zeros to avoid downstream numeric issues
    if not np.isfinite(train_X).all():
        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler.fit(train_X)
    # protect against zero-variance features which make StandardScaler divide-by-zero
    try:
        if hasattr(scaler, 'scale_'):
            scale = getattr(scaler, 'scale_')
            # replace NaN/inf or zero scales with 1.0 to avoid divide-by-zero
            bad = (~np.isfinite(scale)) | (scale == 0)
            if np.any(bad):
                scale = np.array(scale, dtype=np.float64, copy=True)
                scale[~np.isfinite(scale)] = 1.0
                scale[scale == 0] = 1.0
                scaler.scale_ = scale
    except Exception:
        pass

    # Diagnostic: check targets and features in training set for non-finite values
    try:
        # ensure save_dir exists for diagnostic outputs
        os.makedirs(args.save_dir, exist_ok=True)
        # check forward_returns and risk_free_rate in train_df
        fr_vals = train_df['forward_returns'].values if 'forward_returns' in train_df.columns else np.array([])
        rf_vals = train_df['risk_free_rate'].values if 'risk_free_rate' in train_df.columns else np.array([])
        bad_mask = np.zeros(len(train_df), dtype=bool)
        if fr_vals.size > 0:
            bad_mask = bad_mask | (~np.isfinite(fr_vals))
        if rf_vals.size > 0:
            bad_mask = bad_mask | (~np.isfinite(rf_vals))
        # also check feature columns for non-finite
        try:
            feat_arr = train_df[feature_cols].values
            bad_mask = bad_mask | (~np.isfinite(feat_arr).all(axis=1))
        except Exception:
            pass
        if np.any(bad_mask):
            problem_rows = train_df.loc[bad_mask]
            diag_path = os.path.join(args.save_dir, 'diagnostic_problem_rows.csv')
            try:
                problem_rows.to_csv(diag_path, index=False)
                print(f"Diagnostic: found non-finite values in training set, saved rows to {diag_path}")
                if progress_hook:
                    try:
                        progress_hook(f"Diagnostic: found non-finite values in training set, saved rows to {diag_path}")
                    except Exception:
                        pass
            except Exception:
                print("Diagnostic: failed to write diagnostic_problem_rows.csv")
        # sanitize train_df and val_df targets/feature columns to replace non-finite according to strategy
        try:
            if strategy == 'ffill':
                train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
                val_df[feature_cols] = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
            elif strategy == 'zero' or strategy == '0' or strategy == 'none':
                train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                val_df[feature_cols] = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            elif strategy == 'interpolate' or strategy == 'interp':
                train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).interpolate().fillna(train_df[feature_cols].median())
                val_df[feature_cols] = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).interpolate().fillna(val_df[feature_cols].median())
            else:
                train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(train_df[feature_cols].median())
                val_df[feature_cols] = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(val_df[feature_cols].median())
        except Exception:
            pass
        try:
            if 'forward_returns' in train_df.columns:
                train_df['forward_returns'] = train_df['forward_returns'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                val_df['forward_returns'] = val_df['forward_returns'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if 'risk_free_rate' in train_df.columns:
                train_df['risk_free_rate'] = train_df['risk_free_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                val_df['risk_free_rate'] = val_df['risk_free_rate'].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        except Exception:
            pass
    except Exception:
        pass

    # 数据集和加载器
    if args.model == 'lstm':
        train_ds = TimeSeriesDataset(train_df, feature_cols, seq_len=args.seq_len, step=args.train_step, scaler=scaler)
        val_ds = TimeSeriesDataset(val_df, feature_cols, seq_len=args.seq_len, step=args.seq_len, scaler=scaler)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        train_ds = FinancialDataset(train_df, feature_cols)
        val_ds = FinancialDataset(val_df, feature_cols)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 模型
    if args.model == 'lstm':
        model = SimpleLSTMPolicy(input_dim=len(feature_cols), hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout).to(device)
    else:
        # parse mlp hidden layers from args.mlp_hidden (comma-separated)
        try:
            hidden_layers = [int(x) for x in str(getattr(args, 'mlp_hidden', '128,64')).split(',') if x.strip()]
            if len(hidden_layers) == 0:
                hidden_layers = [128, 64]
        except Exception:
            hidden_layers = [128, 64]
        model = SimpleMLP(input_dim=len(feature_cols), hidden_layers=hidden_layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val = -1e9
    best_path = os.path.join(args.save_dir, 'best_model_merged.pth')
    os.makedirs(args.save_dir, exist_ok=True)
    best_positions = None
    best_fr_vals = None
    best_rf_vals = None
    best_date_ids = None

    # 可选的预训练（监督）
    if args.pretrain_epochs > 0:
        print("Stage 1: supervised pretrain (predict forward_returns proxy)")
        if progress_hook:
            try:
                progress_hook(f"Stage 1: supervised pretrain for {args.pretrain_epochs} epochs")
            except Exception:
                pass
        mse = nn.MSELoss()
        for e in range(args.pretrain_epochs):
            model.train()
            losses = []
            for batch in train_loader:
                if args.model == 'lstm':
                    feats = batch['features'].to(device)
                    fr = batch['forward_returns'].to(device)
                    optimizer.zero_grad()
                    raw = model(feats)
                    pred_pos = 2.0 * torch.sigmoid(raw)  # (B, T)
                    # 使用tanh(fr)作为代理（可选地替换market_forward_excess_returns）
                    target = torch.tanh(fr)
                    loss = mse(pred_pos, target)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                else:
                    feats, fr, rf = batch
                    feats = feats.to(device)
                    fr = fr.to(device)
                    optimizer.zero_grad()
                    pred_pos = model(feats)  # (B,)
                    target = torch.tanh(fr)
                    loss = mse(pred_pos, target)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
            # 完整时间序列验证
            report, _, _, _, _ = evaluate_model_full_series_and_report(model, val_df, feature_cols, scaler, device, seq_len=args.seq_len, train_df=train_df, model_type=args.model, visualize=False)
            msg = f"Pretrain Epoch {e+1}/{args.pretrain_epochs} train_loss {np.mean(losses):.6e} val_full_adj_sharpe {report['val_score']:.6f}"
            print(msg)
            if progress_hook:
                try:
                    progress_hook(msg)
                except Exception:
                    pass

    # 第二阶段：使用adjusted_sharpe_loss_double进行微调
    print("Stage 2: fine-tune with AdjustedSharpeLoss (full-series validation)")
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            if args.model == 'lstm':
                feats = batch['features'].to(device)   # (B, T, D)
                fr = batch['forward_returns'].to(device)
                rf = batch['risk_free'].to(device)
                optimizer.zero_grad()
                raw = model(feats)                     # (B, T)
                loss, _ = adjusted_sharpe_loss_via_losspy(raw, fr, rf, lambda_turnover=args.lambda_turnover, l2_reg=args.l2_reg, model=model)
                # detect non-finite loss and save diagnostics
                if not torch.isfinite(loss):
                    try:
                        dpath = os.path.join(args.save_dir, f'diagnostic_nan_loss_epoch{epoch}.npz')
                        os.makedirs(args.save_dir, exist_ok=True)
                        np.savez(dpath, feats=feats.detach().cpu().numpy(), fr=fr.detach().cpu().numpy(), rf=rf.detach().cpu().numpy())
                        print(f"Diagnostic: NaN loss detected, batch saved to {dpath}")
                        if progress_hook:
                            try:
                                progress_hook(f"Diagnostic: NaN loss detected, batch saved to {dpath}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    raise RuntimeError('NaN loss encountered during training')
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                batch_losses.append(float(loss.item()))
            else:
                feats, fr, rf = batch
                feats = feats.to(device)
                fr = fr.to(device)
                rf = rf.to(device)
                optimizer.zero_grad()
                preds = model(feats)  # (B,)
                loss, _ = adjusted_sharpe_loss_via_losspy(preds, fr, rf, lambda_turnover=args.lambda_turnover, l2_reg=args.l2_reg, model=model)
                if not torch.isfinite(loss):
                    try:
                        dpath = os.path.join(args.save_dir, f'diagnostic_nan_loss_epoch{epoch}.npz')
                        os.makedirs(args.save_dir, exist_ok=True)
                        np.savez(dpath, feats=feats.detach().cpu().numpy(), fr=fr.detach().cpu().numpy(), rf=rf.detach().cpu().numpy(), preds=preds.detach().cpu().numpy())
                        print(f"Diagnostic: NaN loss detected, batch saved to {dpath}")
                        if progress_hook:
                            try:
                                progress_hook(f"Diagnostic: NaN loss detected, batch saved to {dpath}")
                            except Exception:
                                pass
                    except Exception:
                        pass
                    raise RuntimeError('NaN loss encountered during training')
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                batch_losses.append(float(loss.item()))

        avg_loss = np.mean(batch_losses) if len(batch_losses) > 0 else float('nan')

        # 完整时间序列验证
        report, date_ids, positions, fr_vals, rf_vals = evaluate_model_full_series_and_report(
            model, val_df, feature_cols, scaler, device, seq_len=args.seq_len, train_df=train_df, model_type=args.model,
            visualize=args.visualize and (epoch % args.viz_every == 0), viz_prefix=f'val_epoch{epoch}', save_dir=args.save_dir
        )
        val_score = report['val_score']
        msg = f"Epoch {epoch}/{args.epochs} - train_loss {avg_loss:.6f} - val_full_adj_sharpe {val_score:.6f} (best {best_val:.6f})"
        print(msg)
        if progress_hook:
            try:
                progress_hook(msg)
            except Exception:
                pass
        scheduler.step(val_score)

        if val_score > best_val:
            best_val = val_score
            # include model architecture info so external loaders can reconstruct
            model_cfg = {}
            if args.model == 'lstm':
                model_cfg = {'input_dim': len(feature_cols), 'hidden_dim': args.hidden_dim, 'n_layers': args.n_layers, 'dropout': args.dropout}
            else:
                # include mlp hidden configuration when available
                try:
                    mlp_hidden = [int(x) for x in str(getattr(args, 'mlp_hidden', '128,64')).split(',') if x.strip()]
                except Exception:
                    mlp_hidden = [128,64]
                model_cfg = {'input_dim': len(feature_cols), 'mlp_hidden': mlp_hidden}
            # atomic, robust save: avoid using torch.save (can trigger native crashes on some platforms)
            # Instead serialize state_dict tensors to numpy arrays and pickle the object atomically.
            state = model.state_dict()
            serial_state = {}
            try:
                for k, v in state.items():
                    try:
                        serial_state[k] = v.detach().cpu().numpy()
                    except Exception:
                        # fallback: try to convert via cpu()
                        serial_state[k] = v.cpu().numpy()
            except Exception:
                serial_state = {k: v for k, v in state.items()}
            save_obj = {'model_state': serial_state, 'scaler': scaler, 'feature_cols': feature_cols, 'model_type': args.model, 'model_cfg': model_cfg}
            # write via pickle atomically
            try:
                d = os.path.dirname(best_path) or '.'
                os.makedirs(d, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmpf:
                    tmpname = tmpf.name
                try:
                    with open(tmpname, 'wb') as f:
                        pickle.dump(save_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                    os.replace(tmpname, best_path)
                    ok, err = True, None
                except Exception as e:
                    try:
                        if os.path.exists(tmpname):
                            os.remove(tmpname)
                    except Exception:
                        pass
                    ok, err = False, e
            except Exception as e:
                ok, err = False, e
            if not ok:
                # warn and continue; do not allow a failed save to crash the process
                try:
                    print(f"Warning: failed to save checkpoint to {best_path}: {err}")
                    if progress_hook:
                        try:
                            progress_hook(f"Warning: failed to save checkpoint: {err}")
                        except Exception:
                            pass
                except Exception:
                    pass
            # record best validation predictions for KPI calculation
            try:
                best_positions = positions
                best_fr_vals = fr_vals
                best_rf_vals = rf_vals
                best_date_ids = date_ids
            except Exception:
                best_positions = None
                best_fr_vals = None
                best_rf_vals = None
            if progress_hook:
                try:
                    progress_hook(f"Saved best model epoch {epoch} -> {best_path}")
                except Exception:
                    pass
            print(f"Saved best model (epoch {epoch}) -> {best_path}")
            # Emit METRIC for GUI orchestrator: indicate a new best model was saved
            try:
                metric = {'event': 'best_model_saved', 'epoch': int(epoch), 'val_score': float(val_score), 'best_path': best_path}
                print('METRIC ' + json.dumps(metric), flush=True)
            except Exception:
                pass
            # 保存诊断
            try:
                plot_positions_and_returns(date_ids, positions, fr_vals, rf_vals, title=f'best_val_epoch{epoch}_adjsharpe{val_score:.4f}', savepath=os.path.join(args.save_dir, f'best_val_epoch{epoch}.png'))
                plot_turnover_hist(positions, savepath=os.path.join(args.save_dir, f'best_val_turnover_epoch{epoch}.png'))
            except Exception:
                pass

    print("Training finished. Best val_full_adj_sharpe:", best_val)
    print("Best model saved to:", best_path)
    if progress_hook:
        try:
            progress_hook(f"Training finished. Best val_full_adj_sharpe: {best_val}")
        except Exception:
            pass

    # compute KPIs for best validation predictions if available
    kpis = None
    try:
        if best_positions is not None and best_fr_vals is not None and best_rf_vals is not None and len(best_positions) > 0:
            adj_sharpe, stats = compute_adjusted_sharpe_numpy(np.array(best_positions), np.array(best_fr_vals), np.array(best_rf_vals))
            kpis = {'adjusted_sharpe': adj_sharpe, 'stats': stats}
    except Exception:
        kpis = None

    # collect artifact list in save_dir
    artifacts = []
    try:
        for fn in os.listdir(args.save_dir):
            artifacts.append(os.path.join(args.save_dir, fn))
    except Exception:
        artifacts = []

    # 最终测试评估（如果提供）
    if args.test_csv and os.path.exists(args.test_csv):
        print("Evaluating best model on test.csv ...")
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state'])
        saved_scaler = ck.get('scaler', scaler)
        test_df = pd.read_csv(args.test_csv).sort_values('date_id').reset_index(drop=True)
        report, date_ids, positions, fr_vals, rf_vals = evaluate_model_full_series_and_report(
            model, test_df, feature_cols, saved_scaler, device, seq_len=args.seq_len, train_df=train_df, model_type=args.model,
            visualize=args.visualize, viz_prefix='test', save_dir=args.save_dir
        )
        print("Test report:", report)
    else:
        if args.test_csv:
            print(f"test_csv provided but not found: {args.test_csv}")
    # return metadata for external callers
    out = {'best_val': float(best_val), 'best_path': best_path, 'save_dir': args.save_dir, 'kpis': kpis, 'artifacts': artifacts}
    return out
# ---------------------------
# 完整时间序列验证和报告
# ---------------------------
def evaluate_model_full_series_and_report(model, df_val, feature_cols, scaler, device, seq_len, train_df=None, model_type='lstm', visualize=False, viz_prefix='val', save_dir=None):
    """
    完整时间序列验证和报告
    输入：model, df_val, feature_cols, scaler, device, seq_len, train_df=None, model_type='lstm', visualize=False, viz_prefix='val', save_dir=None
    输出：report, date_ids, positions, fr, rf
    """
    date_ids, positions, fr, rf = predict_full_series(model, df_val, feature_cols, scaler, device, seq_len, model_type=model_type, train_history_df=train_df)
    val_score, stats = score_with_fallback(date_ids, positions, fr, rf)
    report = {'val_score': float(val_score), 'val_stats': stats, 'positions_len': len(positions)}
    if visualize:
        title = f"{viz_prefix}: adj_sharpe={report['val_score']:.4f}"
        savepos = os.path.join(save_dir, f'{viz_prefix}_diagnostics.png') if save_dir else None
        plot_positions_and_returns(date_ids, positions, fr, rf, title=title, savepath=savepos)
        plot_turnover_hist(positions, savepath=(os.path.join(save_dir, f'{viz_prefix}_turnover.png') if save_dir else None))
        # feature drift with train_df if provided
        if train_df is not None and save_dir:
            try:
                drift = feature_drift_report(train_df, df_val, feature_cols, topk=30)
                drift.to_csv(os.path.join(save_dir, f'{viz_prefix}_feature_drift.csv'), index=False)
            except Exception:
                pass
    return report, date_ids, positions, fr, rf

# ---------------------------
# 参数解析器和主函数
# ---------------------------
def parse_args():
    """
    参数解析器
    输入：None
    输出：args: 参数
    """
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', type=str, required=True)
    p.add_argument('--test_csv', type=str, default=None)
    p.add_argument('--model', type=str, choices=['lstm', 'mlp'], default='lstm')
    p.add_argument('--seq_len', type=int, default=252)
    p.add_argument('--train_step', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--mlp_hidden', type=str, default='128,64', help='Comma-separated MLP hidden layer sizes')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--pretrain_epochs', type=int, default=5)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--lambda_turnover', type=float, default=1.0)
    p.add_argument('--l2_reg', type=float, default=0.0)
    p.add_argument('--clip_grad_norm', type=float, default=1.0)
    p.add_argument('--save_dir', type=str, default='./models')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--visualize', action='store_true')
    p.add_argument('--viz_every', type=int, default=1)
    return p.parse_args()


def train_from_config(cfg: dict, progress_hook=None):
    """Bridge for external callers (e.g., GUI).
    `cfg` can contain a subset of arguments accepted by `parse_args()`; missing values use defaults.
    If provided, `progress_hook` should be a callable like `lambda msg: ...` and will be called with status strings.
    Returns: dict with keys `best_val`, `best_path`, `save_dir` or None on failure.
    """
    # default argument values (avoid calling parse_args() which requires CLI flags)
    defaults = {
        'train_csv': None,
        'test_csv': None,
        'model': 'lstm',
        'seq_len': 252,
        'train_step': 10,
        'batch_size': 16,
        'hidden_dim': 128,
        'n_layers': 1,
        'dropout': 0.1,
        'mlp_hidden': '128,64',
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 40,
        'pretrain_epochs': 5,
        'val_ratio': 0.1,
        'lambda_turnover': 1.0,
        'l2_reg': 0.0,
        'clip_grad_norm': 1.0,
        'save_dir': './models',
        'seed': 42,
        'no_cuda': False,
        'visualize': False,
        'viz_every': 1
        ,'missing_value_strategy': 'median'
    }
    merged = defaults.copy()
    if cfg:
        # shallow merge; caller expected to supply keys matching argparse names
        merged.update(cfg)
    # convert to Namespace and call train
    import argparse as _argparse
    args = _argparse.Namespace(**merged)
    # ensure save_dir
    if not getattr(args, 'save_dir', None):
        args.save_dir = tempfile.mkdtemp(prefix='tas_run_')
    os.makedirs(args.save_dir, exist_ok=True)
    result = train(args, progress_hook=progress_hook)
    # write run_info into save_dir for RunStore (select fields, ensure JSON serializable)
    try:
        run_dir = args.save_dir
        os.makedirs(run_dir, exist_ok=True)
        ts = int(time.time())
        run_info = {
            'timestamp': ts,
            'cfg': merged,
            'best_val': float(result.get('best_val')) if isinstance(result, dict) and result.get('best_val') is not None else None,
            'best_path': result.get('best_path') if isinstance(result, dict) else None,
            'kpis': result.get('kpis') if isinstance(result, dict) else None,
            'artifacts': result.get('artifacts') if isinstance(result, dict) else None
        }
        run_path = os.path.join(run_dir, f'run_info_{ts}.json')
        with open(run_path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
    except Exception:
        run_path = None
    # result is expected to be a dict with best_path etc.
    if isinstance(result, dict) and 'best_path' in result:
        out = result.copy()
        out['run_info'] = run_path
        return out
    # fallback: try to resolve best_path
    best_path = os.path.join(args.save_dir, 'best_model_merged.pth')
    if os.path.exists(best_path):
        out = {'best_path': best_path, 'save_dir': args.save_dir, 'run_info': run_path}
        return out
    return {'save_dir': args.save_dir, 'run_info': run_path}

if __name__ == '__main__':
    args = parse_args()
    # call via train_from_config to ensure RunStore entry is written and progress_hook can be attached
    cfg = vars(args)
    res = train_from_config(cfg)
    print('train_from_config result:', res)
