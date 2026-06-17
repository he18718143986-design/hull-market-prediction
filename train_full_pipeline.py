# train_full_pipeline.py
"""
End-to-end: supervised pretrain encoder -> automatic transfer -> PPO fine-tune
Requires: train.csv in current folder. Optional: test.csv for final evaluation.
Savepoints:
 - encoder_base.pth
 - ppo_policy_latest.pth  (ActorCritic state_dict)
 - logs: training_logs.csv
"""
import os
import math
import random
import argparse
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# -------------------------
# Config / Hyperparams
# -------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_LEN = 6              # same window used in PPO env
EPISODE_LEN = 252
TURNOVER_COST = 0.002

# Pretrain
PRETRAIN_BATCH = 256
PRETRAIN_LR = 1e-3
PRETRAIN_EPOCHS = 30
PRETRAIN_PATIENCE = 5
HIDDEN = 128
VAL_RATIO = 0.15

# PPO fine-tune
PPO_LR = 3e-4
PPO_EPOCHS = 6
PPO_CLIP = 0.2
PPO_MINIBATCH = 64
GAMMA = 0.99
LAMBDA = 0.95
N_PPO_UPDATES = 200
STEPS_PER_UPDATE = 2048       # increase for real training
FREEZE_BASE_UPDATES = 10      # freeze encoder base for first few PPO updates (optional)
SAVE_DIR = "./checkpoints"

FEATURE_PREFIXES = ("M", "E", "P", "V", "S", "MOM", "D")

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_feature_cols(df):
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

def preprocess_features(df, feature_cols, mean=None, std=None):
    df_proc = df.copy()
    # forward/backfill + fill remaining with 0
    df_proc[feature_cols] = df_proc[feature_cols].ffill().bfill().fillna(0.0)
    X = df_proc[feature_cols].values.astype(np.float32)
    if mean is None or std is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-6] = 1.0
    X_scaled = (X - mean) / std
    df_proc[feature_cols] = X_scaled
    return df_proc, mean, std

# -------------------------
# Dataset for supervised pretrain
# -------------------------
class WindowDataset(Dataset):
    """
    Each sample: (window_flat + prev_pos) -> label forward_returns at index i
    For pretrain prev_pos will be set to 0.0 (no prior action). That matches PPO full obs shape.
    """
    def __init__(self, df_proc: pd.DataFrame, feature_cols, window_len):
        self.df = df_proc.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_len = window_len
        # valid indices where label exists: i from window_len .. len(df)-1
        self.indices = list(range(window_len, len(self.df)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        window = self.df.loc[i - self.window_len : i - 1, self.feature_cols].values  # (window_len, D)
        x_window = window.flatten().astype(np.float32)
        prev_pos = np.array([0.0], dtype=np.float32)  # pretrain: no previous action
        x = np.concatenate([x_window, prev_pos], axis=0)
        y = np.float32(self.df.loc[i, "forward_returns"])
        return x, y

# -------------------------
# Models: Encoder + ActorCritic
# -------------------------
class EncoderForReg(nn.Module):
    def __init__(self, input_dim, hidden=HIDDEN):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        h = self.base(x)
        out = self.head(h).squeeze(-1)
        return out

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=HIDDEN):
        super().__init__()
        # base must match EncoderForReg.base architecture for weight transfer
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(1) * -1.0)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.base(x)
        mean = self.mean_head(h)
        value = self.value_head(h).squeeze(-1)
        return mean, self.log_std.expand_as(mean).squeeze(-1), value

# -------------------------
# PPO Trainer (same as earlier)
# -------------------------
Transition = namedtuple('Transition', ['obs', 'act', 'logp', 'ret', 'adv', 'val'])

class PPO:
    def __init__(self, obs_dim, lr=PPO_LR, clip=PPO_CLIP, epochs=PPO_EPOCHS, minibatch=PPO_MINIBATCH, gamma=GAMMA, lam=LAMBDA):
        self.ac = ActorCritic(obs_dim).to(DEVICE)
        self.optim = optim.Adam(self.ac.parameters(), lr=lr, weight_decay=1e-4)
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.gamma = gamma
        self.lam = lam

    def get_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, val = self.ac(obs_t)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()
            logp = dist.log_prob(raw).sum(-1)
            # squash to [0,2]
            action = 2.0 * torch.sigmoid(raw)
        return float(action.cpu().numpy().reshape(-1)[0]), float(logp.cpu().numpy().reshape(-1)[0]), float(val.cpu().numpy().reshape(-1)[0])

    def compute_gae(self, rewards, values):
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values + [0.0], dtype=np.float32)
        advs = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advs[t] = gae
        returns = advs + values[:-1]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return returns, advs

    def update(self, transitions):
        obs = torch.tensor(np.vstack([t.obs for t in transitions]), dtype=torch.float32, device=DEVICE)
        acts = torch.tensor(np.vstack([t.act for t in transitions]), dtype=torch.float32, device=DEVICE)
        old_logp = torch.tensor([t.logp for t in transitions], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        returns = torch.tensor([t.ret for t in transitions], dtype=torch.float32, device=DEVICE)
        advs = torch.tensor([t.adv for t in transitions], dtype=torch.float32, device=DEVICE)

        n = obs.size(0)
        for epoch in range(self.epochs):
            idxs = np.random.permutation(n)
            for start in range(0, n, self.minibatch):
                mb_idx = idxs[start:start + self.minibatch]
                b_obs = obs[mb_idx]
                b_acts = acts[mb_idx]
                b_old_logp = old_logp[mb_idx]
                b_returns = returns[mb_idx]
                b_advs = advs[mb_idx]

                mean, log_std, vals = self.ac(b_obs)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)

                eps = 1e-6
                scaled = torch.clamp(b_acts / 2.0, eps, 1.0 - eps)
                raw = torch.log(scaled) - torch.log1p(-scaled)

                logp = dist.log_prob(raw).sum(-1, keepdim=True)
                ratio = torch.exp(logp - b_old_logp)

                surr1 = ratio * b_advs.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * b_advs.unsqueeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(vals, b_returns)
                entropy_bonus = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optim.step()

# -------------------------
# MarketEnv for PPO
# -------------------------
class MarketEnv:
    def __init__(self, data: pd.DataFrame, feature_cols, window_len=WINDOW_LEN, episode_len=EPISODE_LEN, turnover_cost=TURNOVER_COST):
        assert 'forward_returns' in data.columns and 'risk_free_rate' in data.columns
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_len = window_len
        self.episode_len = episode_len
        self.turnover_cost = turnover_cost
        self.max_start = len(self.data) - (self.window_len + self.episode_len) - 1
        if self.max_start <= 0:
            raise ValueError("Data too short for window+episode length")
        self.reset()

    def reset(self, random_start=True):
        self.start_idx = np.random.randint(0, self.max_start) if random_start else 0
        self.t = 0
        self.idx = self.start_idx + self.window_len
        self.prev_pos = 0.0
        self.episode_positions = []
        self.episode_rewards = []
        return self._get_obs()

    def _get_obs(self):
        window = self.data.loc[self.idx - self.window_len : self.idx - 1, self.feature_cols].values
        obs = np.concatenate([window.flatten(), np.array([self.prev_pos], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def step(self, action: float):
        pos = float(np.clip(action, 0.0, 2.0))
        row = self.data.loc[self.idx]
        rf = float(row['risk_free_rate'])
        fwd = float(row['forward_returns'])

        strat_return = rf * (1.0 - pos) + pos * fwd
        excess = strat_return - rf
        turnover_pen = self.turnover_cost * abs(pos - self.prev_pos)
        reward = excess - turnover_pen

        self.episode_positions.append(pos)
        self.episode_rewards.append(reward)
        self.prev_pos = pos

        self.t += 1
        self.idx += 1
        done = (self.t >= self.episode_len)
        obs = self._get_obs() if not done else None
        info = {}

        if done:
            term_bonus = self._terminal_adjusted_sharpe_bonus()
            reward += term_bonus
            info['terminal_adjusted_sharpe_bonus'] = term_bonus

        return obs, float(reward), done, info

    def _terminal_adjusted_sharpe_bonus(self):
        positions = np.array(self.episode_positions, dtype=np.float32)
        idxs = np.arange(self.start_idx + self.window_len, self.start_idx + self.window_len + self.episode_len)
        fr = self.data.loc[idxs, 'forward_returns'].values.astype(np.float32)
        rf = self.data.loc[idxs, 'risk_free_rate'].values.astype(np.float32)
        strat = rf * (1.0 - positions) + positions * fr
        strat_excess = strat - rf

        eps = 1e-8
        se = np.clip(strat_excess, -1 + eps, None)
        me = np.clip(fr - rf, -1 + eps, None)

        strategy_geo = np.exp(np.mean(np.log1p(se))) - 1.0
        market_geo   = np.exp(np.mean(np.log1p(me))) - 1.0

        strategy_std = np.std(strat, ddof=0) + eps
        market_std   = np.std(fr, ddof=0) + eps

        trading_days = 252.0
        strategy_vol = float(strategy_std * math.sqrt(trading_days) * 100.0)
        market_vol   = float(market_std * math.sqrt(trading_days) * 100.0)

        sharpe = (strategy_geo / (strategy_std + eps)) * math.sqrt(trading_days)

        excess_vol = max(0.0, (strategy_vol / (market_vol + eps)) - 1.2) if market_vol > 0 else 0.0
        vol_penalty = 1.0 + excess_vol

        return_gap = max(0.0, (market_geo - strategy_geo)) * 100.0 * trading_days
        return_penalty = 1.0 + (return_gap ** 2) / 100.0

        adjusted_sharpe = sharpe / (vol_penalty * return_penalty + eps)
        bonus = float(np.clip(adjusted_sharpe, -5.0, 5.0))
        return bonus

    def render(self):
        return pd.DataFrame({'position': self.episode_positions, 'reward': self.episode_rewards})

# -------------------------
# Score-like fallback (if user has score.py, that will be used)
# -------------------------
def score_like(solution_df: pd.DataFrame, submission_df: pd.DataFrame):
    # minimal in-script reimplementation of contest score (expects same columns)
    sol = solution_df.copy()
    sol['position'] = submission_df['allocation'].values
    MIN_INVESTMENT = 0
    MAX_INVESTMENT = 2
    if sol['position'].max() > MAX_INVESTMENT or sol['position'].min() < MIN_INVESTMENT:
        raise ValueError("position out of bounds")
    sol['strategy_returns'] = sol['risk_free_rate'] * (1 - sol['position']) + sol['position'] * sol['forward_returns']
    strategy_excess = sol['strategy_returns'] - sol['risk_free_rate']
    strategy_geo = (np.prod(1 + strategy_excess)) ** (1 / len(sol)) - 1
    strategy_std = sol['strategy_returns'].std()
    trading_days = 252.0
    if strategy_std == 0:
        return 0.0
    sharpe = strategy_geo / (strategy_std + 1e-8) * math.sqrt(trading_days)
    market_excess = sol['forward_returns'] - sol['risk_free_rate']
    market_geo = (np.prod(1 + market_excess)) ** (1 / len(sol)) - 1
    market_std = sol['forward_returns'].std() + 1e-8
    strategy_vol = float(strategy_std * math.sqrt(trading_days) * 100.0)
    market_vol = float(market_std * math.sqrt(trading_days) * 100.0)
    excess_vol = max(0, strategy_vol / (market_vol + 1e-8) - 1.2) if market_vol > 0 else 0
    vol_penalty = 1 + excess_vol
    return_gap = max(0, (market_geo - strategy_geo) * 100 * trading_days)
    return_penalty = 1 + (return_gap ** 2) / 100.0
    adjusted = sharpe / (vol_penalty * return_penalty)
    return adjusted

# -------------------------
# Main flow: pretrain -> transfer -> PPO
# -------------------------
def main(train_csv="train.csv", test_csv="test.csv"):
    set_seed(SEED)
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} not found")

    df = pd.read_csv(train_csv)
    df = df.sort_values("date_id").reset_index(drop=True)
    feature_cols = select_feature_cols(df)
    if not feature_cols:
        raise ValueError("No feature columns found. Check prefixes or train.csv")

    # preprocess
    df_proc, feat_mean, feat_std = preprocess_features(df, feature_cols)

    # supervised pretrain
    print("=== Supervised pretraining encoder (predict forward_returns) ===")
    dataset = WindowDataset(df_proc, feature_cols, WINDOW_LEN)
    N = len(dataset)
    val_n = max(1, int(N * VAL_RATIO))
    train_n = N - val_n
    train_ds = Subset(dataset, range(0, train_n))
    val_ds = Subset(dataset, range(train_n, N))

    train_loader = DataLoader(train_ds, batch_size=PRETRAIN_BATCH, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=PRETRAIN_BATCH, shuffle=False, drop_last=False)

    input_dim = len(feature_cols) * WINDOW_LEN + 1  # +1 for prev_pos
    encoder = EncoderForReg(input_dim, hidden=HIDDEN).to(DEVICE)
    opt = optim.AdamW(encoder.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience = 0
    encoder_save = os.path.join(SAVE_DIR, "encoder_base.pth")

    for ep in range(1, PRETRAIN_EPOCHS + 1):
        encoder.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = encoder(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0); opt.step()
            total_loss += float(loss.item()) * xb.size(0); seen += xb.size(0)
        train_loss = total_loss / max(1, seen)

        # val
        encoder.eval()
        vloss = 0.0; vseen = 0
        mae = 0.0; dir_corr = 0; tot=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = encoder(xb)
                loss = criterion(pred, yb)
                vloss += float(loss.item()) * xb.size(0); vseen += xb.size(0)
                mae += float(torch.mean(torch.abs(pred-yb)).item()) * xb.size(0)
                dir_corr += int(((pred * yb) > 0).sum().item()); tot += xb.size(0)
        val_loss = vloss / max(1, vseen); val_mae = mae / max(1, vseen); dir_acc = dir_corr / max(1, tot)
        print(f"Pretrain Ep {ep:03d} | train_mse {train_loss:.6e} | val_mse {val_loss:.6e} | val_mae {val_mae:.6e} | dir_acc {dir_acc:.4f}")

        sched.step(val_loss)
        if val_loss < best_val - 1e-9:
            best_val = val_loss; patience = 0
            torch.save(encoder.base.state_dict(), encoder_save)
            print(f"  -> saved encoder base to {encoder_save}")
        else:
            patience += 1
            if patience >= PRETRAIN_PATIENCE:
                print("Early stopping pretrain")
                break

    # Transfer to PPO
    print("=== Transfer encoder -> PPO ActorCritic.base ===")
    env = MarketEnv(df_proc, feature_cols=feature_cols, window_len=WINDOW_LEN, episode_len=EPISODE_LEN, turnover_cost=TURNOVER_COST)
    obs_dim = len(feature_cols) * WINDOW_LEN + 1
    ppo = PPO(obs_dim)
    # load encoder base state into ac.base
    enc_state = torch.load(encoder_save, map_location=DEVICE)
    # try loading; if mismatch shape, raise informative error
    try:
        ppo.ac.base.load_state_dict(enc_state)
        print("Loaded encoder weights into ActorCritic.base")
    except Exception as e:
        print("Failed to load encoder base weights into ActorCritic.base:", e)
        print("Ensure base architectures match. Aborting.")
        return

    # Optionally freeze base for first FREEZE_BASE_UPDATES updates
    base_params = list(ppo.ac.base.parameters())
    def set_base_requires_grad(flag):
        for p in base_params:
            p.requires_grad = flag

    # Evaluate hook: try to import contest score
    score_fn = None
    try:
        from score import score as contest_score
        score_fn = contest_score
        print("Using contest score() from score.py for test evaluation")
    except Exception:
        print("score.py not found or import failed; using local score_like() for final evaluation")
        score_fn = score_like

    # PPO training
    print("=== Starting PPO fine-tune ===")
    transitions_all = []
    best_test_score = -1e9
    policy_save = os.path.join(SAVE_DIR, "ppo_policy_latest.pth")
    logs = []
    for update in range(1, N_PPO_UPDATES + 1):
        # freeze base initially if required
        if update == 1 and FREEZE_BASE_UPDATES > 0:
            set_base_requires_grad(False)
            print(f"Freezing base for first {FREEZE_BASE_UPDATES} updates")
        if update == FREEZE_BASE_UPDATES + 1 and FREEZE_BASE_UPDATES > 0:
            set_base_requires_grad(True)
            print("Unfreezing base parameters")

        transitions = []
        steps_collected = 0
        ep_returns = []
        positions_roll = []
        turnovers_roll = []
        while steps_collected < STEPS_PER_UPDATE:
            obs = env.reset(random_start=True)
            done = False
            ep_obs, ep_acts, ep_logp, ep_vals, ep_rews = [], [], [], [], []
            while not done and steps_collected < STEPS_PER_UPDATE:
                action, logp, val = ppo.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                ep_obs.append(obs.copy()); ep_acts.append([action]); ep_logp.append(logp); ep_vals.append(val); ep_rews.append(reward)
                obs = next_obs if next_obs is not None else np.zeros_like(ep_obs[-1])
                steps_collected += 1
            returns, advs = ppo.compute_gae(ep_rews, ep_vals)
            for o,a,lp,ret,adv,v in zip(ep_obs, ep_acts, ep_logp, returns, advs, ep_vals):
                transitions.append(Transition(o, a, lp, float(ret), float(adv), float(v)))
            ep_returns.append(sum(ep_rews))
            positions_roll.append(np.mean(ep_acts))
            # approximate turnover
            pos_seq = np.array([a[0] for a in ep_acts])
            if len(pos_seq) > 1:
                turnovers_roll.append(np.mean(np.abs(np.diff(pos_seq))))
        # update
        ppo.update(transitions)

        avg_ret = np.mean(ep_returns)
        avg_pos = float(np.mean(positions_roll)) if positions_roll else 0.0
        avg_turnover = float(np.mean(turnovers_roll)) if turnovers_roll else 0.0
        logs.append({"update": update, "avg_ret": avg_ret, "avg_pos": avg_pos, "avg_turnover": avg_turnover})
        print(f"Update {update}/{N_PPO_UPDATES} | avg_ret {avg_ret:.6f} | avg_pos {avg_pos:.4f} | avg_turnover {avg_turnover:.6f}")

        # periodic save
        if update % 10 == 0:
            torch.save(ppo.ac.state_dict(), policy_save)
            pd.DataFrame(logs).to_csv(os.path.join(SAVE_DIR, "training_logs.csv"), index=False)

    # final save
    torch.save(ppo.ac.state_dict(), policy_save)
    pd.DataFrame(logs).to_csv(os.path.join(SAVE_DIR, "training_logs.csv"), index=False)
    print("PPO training finished. Policy saved to:", policy_save)

    # Evaluate on test.csv or trailing portion of train
    if os.path.exists(test_csv):
        print("Evaluating on test.csv using policy (deterministic mean action).")
        df_test = pd.read_csv(test_csv).sort_values("date_id").reset_index(drop=True)
        df_test_proc, _, _ = preprocess_features(df_test, feature_cols, mean=feat_mean, std=feat_std)
        positions = []
        fr_list = []
        rf_list = []
        did_list = []
        prev_pos = 0.0
        n = len(df_test_proc)
        if n <= WINDOW_LEN + 1:
            print("test.csv too short for evaluation")
        else:
            for idx in range(WINDOW_LEN, n):
                window = df_test_proc.loc[idx - WINDOW_LEN: idx - 1, feature_cols].values
                obs = np.concatenate([window.flatten(), np.array([prev_pos], dtype=np.float32)], axis=0).astype(np.float32)
                with torch.no_grad():
                    mean, log_std, val = ppo.ac(torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0))
                    raw = mean[0,0]; action = 2.0 * torch.sigmoid(raw)
                    pos = float(action.cpu().item())
                row = df_test_proc.loc[idx]
                positions.append(pos); fr_list.append(float(row['forward_returns'])); rf_list.append(float(row['risk_free_rate'])); did_list.append(int(row['date_id']))
                prev_pos = pos
            solution = pd.DataFrame({"date_id": did_list, "forward_returns": fr_list, "risk_free_rate": rf_list, "market_forward_excess_returns": np.array(fr_list)-np.array(rf_list)})
            submission = pd.DataFrame({"date_id": did_list, "allocation": np.array(positions)})
            # try contest score
            try:
                from score import score as contest_score
                test_score = contest_score(solution, submission, row_id_column_name="date_id")
            except Exception:
                test_score = score_like(solution, submission)
            print("="*50)
            print("Final evaluation on test.csv: adjusted_sharpe =", test_score)
            print("="*50)
    else:
        print("No test.csv: you can evaluate policy by running evaluate_on_test() with your own data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    args = parser.parse_args()
    main(train_csv=args.train, test_csv=args.test)
