# supervised_pretrain.py
# 监督预训练 encoder（predict forward_returns）
#
# Usage:
#   python supervised_pretrain.py
#
# 输出: encoder_pretrained.pth (包含 encoder.base 的 state_dict)
# 然后在你的 PPO 脚本中：
#   from ppo_market_agent import ActorCritic
#   ac = ActorCritic(obs_dim)
#   encoder_state = torch.load('encoder_pretrained.pth')
#   ac.base.load_state_dict(encoder_state)

import os
import math
import numpy as np
import pandas as pd
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random

# -------------------------
# Config
# -------------------------
TRAIN_CSV = "train.csv"
SAVE_PATH = "encoder_pretrained.pth"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_LEN = 6            # 与 PPO 环境一致的 window_len
HIDDEN = 128
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 6             # early stopping patience on val loss
VAL_RATIO = 0.15

# Optional: try to import helper functions and ActorCritic from ppo_market_agent.py
try:
    from ppo_market_agent import select_feature_cols, preprocess_features, ActorCritic
    _IMPORTED_FROM_PPO = True
except Exception:
    _IMPORTED_FROM_PPO = False

# If not imported, define fallback select_feature_cols & preprocess_features
FEATURE_PREFIXES = ("M", "E", "P", "V", "S", "MOM", "D")
def select_feature_cols_fallback(df: pd.DataFrame) -> List[str]:
    drop_cols = {"date_id", "forward_returns", "risk_free_rate", "market_forward_excess_returns"}
    feature_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        for p in FEATURE_PREFIXES:
            if c.startswith(p):
                feature_cols.append(c)
                break
    return feature_cols

def preprocess_features_fallback(df: pd.DataFrame, feature_cols, mean=None, std=None):
    df_proc = df.copy()
    df_proc[feature_cols] = df_proc[feature_cols].ffill().bfill().fillna(0.0)
    X = df_proc[feature_cols].values.astype(np.float32)
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
    X_scaled = (X - mean) / std
    df_proc[feature_cols] = X_scaled
    return df_proc, mean, std

# Dataset: sliding windows -> label (next-day forward_returns)
class WindowDataset(Dataset):
    def __init__(self, df_proc: pd.DataFrame, feature_cols: List[str], window_len: int):
        self.feature_cols = feature_cols
        self.window_len = window_len
        self.df = df_proc.reset_index(drop=True)
        # start at index window_len, label is forward_returns at idx
        self.indices = list(range(window_len, len(self.df)))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        # features from i-window_len ... i-1
        window = self.df.loc[i - self.window_len : i - 1, self.feature_cols].values  # (window_len, D)
        x = window.flatten().astype(np.float32)  # shape (window_len * D,)
        y = np.float32(self.df.loc[i, "forward_returns"])
        return x, y

# Encoder model (same structure as ActorCritic.base)
class EncoderForReg(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden, 1)  # predict scalar forward_returns
    def forward(self, x):
        h = self.base(x)
        out = self.head(h).squeeze(-1)
        return out

# Optional: if ActorCritic was imported, we can later copy base weights into ActorCritic
# ActorCritic should define base = nn.Sequential(...) with same structure

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train():
    set_seed(SEED)
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"{TRAIN_CSV} not found in current dir. Place your train.csv here.")

    df = pd.read_csv(TRAIN_CSV)
    df = df.sort_values("date_id").reset_index(drop=True)

    # choose feature cols
    if _IMPORTED_FROM_PPO:
        feature_cols = select_feature_cols(df)
        df_proc, feat_mean, feat_std = preprocess_features(df, feature_cols)
    else:
        feature_cols = select_feature_cols_fallback(df)
        df_proc, feat_mean, feat_std = preprocess_features_fallback(df, feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Check prefixes or train.csv columns.")

    # build dataset (time-ordered)
    dataset = WindowDataset(df_proc, feature_cols, window_len=WINDOW_LEN)
    N = len(dataset)
    val_n = max(1, int(N * VAL_RATIO))
    train_n = N - val_n
    # split by index to keep time order (train first, val last)
    train_indices = list(range(0, train_n))
    val_indices = list(range(train_n, N))

    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = len(feature_cols) * WINDOW_LEN
    model = EncoderForReg(input_dim, hidden=HIDDEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 0

    for ep in range(1, EPOCHS + 1):
        # train epoch
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / max(1, n_seen)

        # val
        model.eval()
        v_loss = 0.0
        n_val = 0
        # extra metrics: MAE and directional acc
        mae_sum = 0.0
        dir_correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                v_loss += float(loss.item()) * xb.size(0)
                n_val += xb.size(0)
                mae_sum += float(torch.mean(torch.abs(pred - yb)).item()) * xb.size(0)
                # directional accuracy
                dir_correct += int(((pred * yb) > 0).sum().item())  # both >0 or both <0 counts
                total += xb.size(0)
        val_loss = v_loss / max(1, n_val)
        val_mae = mae_sum / max(1, n_val)
        dir_acc = dir_correct / max(1, total)

        print(f"Epoch {ep:03d} | train_mse: {train_loss:.6e} | val_mse: {val_loss:.6e} | val_mae: {val_mae:.6e} | dir_acc: {dir_acc:.4f}")

        scheduler.step(val_loss)
        # early stopping
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            patience = 0
            # save encoder base weights (state_dict of model.base)
            # We'll save only base to be loaded into ActorCritic.base
            torch.save(model.base.state_dict(), SAVE_PATH)
            print(f"  -> new best val {best_val:.6e}, saved base to {SAVE_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {ep}; best_val {best_val:.6e}")
                break

    # End of training
    print("Pretraining finished. Saved base state_dict at:", SAVE_PATH)
    print("To load into ActorCritic (in your PPO code):")
    print("    from ppo_market_agent import ActorCritic")
    print(f"    ac = ActorCritic(obs_dim={input_dim + 1})  # +1 for prev_pos if your ActorCritic expects that\n    ac.base.load_state_dict(torch.load('{SAVE_PATH}'))")
    # Note: obs_dim in ActorCritic is full obs (window flattened + prev_pos), but our encoder trained on window only.
    # If your ActorCritic.base expects the full obs dimension, you may want to re-train encoder including prev_pos
    # or initialize ac.base with the matching shapes (we trained base for flattened window only).

if __name__ == "__main__":
    train()
