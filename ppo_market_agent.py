# ppo_market_agent.py
# 最小 PPO + 市场环境 for "Hull Tactical" 风格数据
# 保存并本地运行. 替换合成生成器 with your train.csv 预处理.

import math
import os
import numpy as np
import pandas as pd
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from score import score  # 与竞赛一致的 Adjusted Sharpe 评分函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可选特征前缀：根据实际列名调整
FEATURE_PREFIXES = ("M", "E", "P", "V", "S", "MOM", "D")


def select_feature_cols(df: pd.DataFrame):
    """
    从 DataFrame 中选择作为模型输入的特征列：
    - 排除标签 / 环境列（date_id, forward_returns, risk_free_rate, market_forward_excess_returns）
    - 仅保留以指定前缀开头的列（M*, E*, P*, V*, S*, MOM*, D*）
    """
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


def preprocess_features(df: pd.DataFrame, feature_cols, mean=None, std=None):
    """
    对特征列做缺失值填充和标准化：
    - 先向前填充，再用 0 填充剩余 NaN
    - 如果未提供 mean/std，则在当前 df 上计算；否则使用传入的统计量（用于 test）
    返回：预处理后的 df 副本、feature_mean、feature_std
    """
    df_proc = df.copy()
    # 缺失值处理
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
# 市场环境
# -------------------------
class MarketEnv:
    """
    Gym-like environment (no gym dependency).
    - data: DataFrame with 'forward_returns' and 'risk_free_rate' and feature columns.
    - feature_cols: list of feature column names used as observation input.
    - window_len: how many past days of features to include (flattened).
    - episode_len: number of trading days per episode (e.g., 252).
    - turnover_cost: coefficient for per-step turnover penalty (e.g., 0.002).
    Observation: numpy float32 vector: [features(window_len days).flatten(), prev_pos]
    Action: scalar in [0,2] (position). We map network output through sigmoid*2.
    Reward: step_reward = strategy_excess_return - turnover_cost * abs(Δpos).
            Terminal: compute adjusted_sharpe and add as a terminal bonus (clipped).
    """
    def __init__(self, data: pd.DataFrame, feature_cols, window_len=10, episode_len=252, turnover_cost=0.001):
        assert 'forward_returns' in data.columns and 'risk_free_rate' in data.columns
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_len = window_len
        self.episode_len = episode_len
        self.turnover_cost = turnover_cost

        # 起始索引必须允许窗口 + 剧集
        self.max_start = len(self.data) - (self.window_len + self.episode_len) - 1
        if self.max_start <= 0:
            raise ValueError("Data too short for window+episode length")
        self.reset()

    def reset(self, random_start=True):
        self.start_idx = np.random.randint(0, self.max_start) if random_start else 0
        self.t = 0
        self.idx = self.start_idx + self.window_len  # 指向当前数据日的指针
        self.prev_pos = 0.0
        self.episode_positions = []
        self.episode_rewards = []
        return self._get_obs()

    def _get_obs(self):
        # 获取特征窗口 [idx-window_len: idx-1]
        window = self.data.loc[self.idx - self.window_len : self.idx - 1, self.feature_cols].values
        obs = np.concatenate([window.flatten(), np.array([self.prev_pos], dtype=np.float32)], axis=0)
        return obs.astype(np.float32)

    def step(self, action: float):
        # 限制动作并计算收益
        pos = float(np.clip(action, 0.0, 2.0))
        row = self.data.loc[self.idx]
        rf = float(row['risk_free_rate'])
        fwd = float(row['forward_returns'])

        strat_return = rf * (1.0 - pos) + pos * fwd
        excess = strat_return - rf
        turnover_pen = self.turnover_cost * abs(pos - self.prev_pos)
        reward = excess - turnover_pen

        # 记录
        self.episode_positions.append(pos)
        self.episode_rewards.append(reward)
        self.prev_pos = pos

        self.t += 1
        self.idx += 1
        done = (self.t >= self.episode_len)
        obs = self._get_obs() if not done else None
        info = {}

        if done:
            # calculate adjusted_sharpe-like bonus and add to final reward
            term_bonus = self._terminal_adjusted_sharpe_bonus()
            reward += term_bonus
            info['terminal_adjusted_sharpe_bonus'] = term_bonus

        return obs, float(reward), done, info

    def _terminal_adjusted_sharpe_bonus(self):
        """
        Compute a terminal bonus based on the adjusted Sharpe used in the contest:
        - geometric means for returns (via log1p)
        - strategy volatility (std) annualized
        - vol and return penalties as in scoring code
        Return a clipped bonus to keep training stable.
        """
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

        # 波动率惩罚和收益惩罚 (相同的公式 as scoring)
        excess_vol = max(0.0, (strategy_vol / (market_vol + eps)) - 1.2) if market_vol > 0 else 0.0
        vol_penalty = 1.0 + excess_vol

        return_gap = max(0.0, (market_geo - strategy_geo)) * 100.0 * trading_days
        return_penalty = 1.0 + (return_gap ** 2) / 100.0

        adjusted_sharpe = sharpe / (vol_penalty * return_penalty + eps)

        # 缩放 & 裁剪奖励以避免大梯度信号; 根据需要调整缩放
        bonus = float(np.clip(adjusted_sharpe, -5.0, 5.0))
        return bonus

    def render(self):
        return pd.DataFrame({'position': self.episode_positions, 'reward': self.episode_rewards})

# -------------------------
# Actor-Critic 网络
# -------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden, 1)       # 输出原始动作 (无界)
        self.log_std = nn.Parameter(torch.ones(1) * -1.0)  # learnable state-independent log std
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.base(x)
        mean = self.mean_head(h)         # shape (B,1)
        value = self.value_head(h).squeeze(-1)  # shape (B,)
        # log_std 扩展到匹配 mean 形状
        return mean, self.log_std.expand_as(mean).squeeze(-1), value

# -------------------------
# PPO 训练器 (vanilla clipped PPO)
# -------------------------
Transition = namedtuple('Transition', ['obs', 'act', 'logp', 'ret', 'adv', 'val'])

class PPO:
    def __init__(self, obs_dim, lr=3e-4, clip=0.2, epochs=10, minibatch=64, gamma=0.99, lam=0.95):
        self.ac = ActorCritic(obs_dim).to(device)
        self.optim = optim.Adam(self.ac.parameters(), lr=lr, weight_decay=1e-4)
        self.clip = clip
        self.epochs = epochs
        self.minibatch = minibatch
        self.gamma = gamma
        self.lam = lam

    def get_action(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, val = self.ac(obs_t)      # mean shape (1,1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            raw = dist.rsample()   # 重参数化采样
            logp = dist.log_prob(raw).sum(-1)
            # 压缩到 [0,2]
            action = 2.0 * torch.sigmoid(raw)
        # 返回标量
        return float(action.cpu().numpy().reshape(-1)[0]), float(logp.cpu().numpy().reshape(-1)[0]), float(val.cpu().numpy().reshape(-1)[0])

    def compute_gae(self, rewards, values):
        """
        rewards, values are lists for one episode.
        returns: returns array, normalized advantages array
        """
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
        # 组装数组
        obs = torch.tensor(np.vstack([t.obs for t in transitions]), dtype=torch.float32, device=device)
        acts = torch.tensor(np.vstack([t.act for t in transitions]), dtype=torch.float32, device=device)
        old_logp = torch.tensor([t.logp for t in transitions], dtype=torch.float32, device=device).unsqueeze(-1)
        returns = torch.tensor([t.ret for t in transitions], dtype=torch.float32, device=device)
        advs = torch.tensor([t.adv for t in transitions], dtype=torch.float32, device=device)

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

                # 反转 sigmoid 变换: raw = logit(action/2)
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
# 合成数据助手 (用于快速实验)
# -------------------------
def make_synthetic_data(n_days=3000, n_feats=8, seed=42):
    rs = np.random.RandomState(seed)
    shocks = rs.normal(scale=0.01, size=n_days)
    fr = np.zeros(n_days)
    for t in range(1, n_days):
        fr[t] = 0.0002 + 0.98 * fr[t - 1] + shocks[t]
    rf = 0.0001 + rs.normal(scale=1e-5, size=n_days)
    feats = rs.normal(size=(n_days, n_feats))
    df = pd.DataFrame(feats, columns=[f'F{i}' for i in range(n_feats)])
    df['forward_returns'] = fr
    df['risk_free_rate'] = rf
    df['date_id'] = np.arange(n_days)
    return df

# -------------------------
# 示例训练脚本
# -------------------------
def train_example():
    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")

    df = pd.read_csv(train_path)
    df = df.sort_values("date_id").reset_index(drop=True)

    # 选择特征列（按前缀）
    feature_cols = select_feature_cols(df)
    if not feature_cols:
        raise ValueError("未找到以指定前缀开头的特征列，请检查 FEATURE_PREFIXES 和 train.csv 列名。")

    # 预处理训练特征并标准化
    df_proc, feat_mean, feat_std = preprocess_features(df, feature_cols)

    # 使用预处理后的训练数据构建环境
    env = MarketEnv(df_proc, feature_cols=feature_cols, window_len=6, episode_len=252, turnover_cost=0.002)

    obs_dim = len(feature_cols) * env.window_len + 1
    ppo = PPO(obs_dim, lr=3e-4, clip=0.2, epochs=6, minibatch=64, gamma=0.99, lam=0.95)

    n_updates = 200            # 真实实验: 许多更新
    steps_per_update = 2048    # 推荐: 大 rollouts 估计年化指标
    for update in range(n_updates):
        transitions = []
        steps_collected = 0
        ep_returns = []
        while steps_collected < steps_per_update:
            obs = env.reset(random_start=True)
            done = False
            ep_obs, ep_acts, ep_logp, ep_vals, ep_rews = [], [], [], [], []
            while not done and steps_collected < steps_per_update:
                action, logp, val = ppo.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                ep_obs.append(obs.copy())
                ep_acts.append([action])
                ep_logp.append(logp)
                ep_vals.append(val)
                ep_rews.append(reward)
                obs = next_obs if next_obs is not None else np.zeros_like(ep_obs[-1])
                steps_collected += 1
            returns, advs = ppo.compute_gae(ep_rews, ep_vals)
            for o,a,lp,ret,adv,v in zip(ep_obs, ep_acts, ep_logp, returns, advs, ep_vals):
                transitions.append(Transition(o, a, lp, float(ret), float(adv), float(v)))
            ep_returns.append(sum(ep_rews))

        # 更新策略
        ppo.update(transitions)

        # 记录: 在固定起始点上平均剧集返回和评估确定性策略 
        avg_ret = np.mean(ep_returns)
        # 快速评估
        obs_eval = env.reset(random_start=False)
        done=False
        eval_rews=[]
        while not done:
            mean, log_std, val = ppo.ac(torch.tensor(obs_eval, dtype=torch.float32, device=device).unsqueeze(0))
            raw = mean.detach().cpu().numpy()[0,0]
            action = 2.0 * (1.0 / (1.0 + np.exp(-raw)))
            obs_eval, reward, done, info = env.step(action)
            eval_rews.append(reward)
        eval_bonus = info.get('terminal_adjusted_sharpe_bonus', 0.0)
        print(f"Update {update+1}/{n_updates} | avg_collected_episode_return {avg_ret:.6f} | eval_term_bonus {eval_bonus:.4f}")

    # 训练结束后，在 test.csv 上用官方 score() 做最终评估
    if os.path.exists(test_path):
        evaluate_on_test(ppo, test_path, feature_cols, feat_mean, feat_std, window_len=6)
    else:
        print("未找到 test.csv，跳过在测试集上的最终评估。")


def evaluate_on_test(ppo: PPO, test_csv_path: str, feature_cols, feat_mean, feat_std, window_len: int = 6):
    """
    使用训练好的 PPO 策略在 test.csv 上评估：
    - 使用与竞赛一致的 score() 函数，计算 Adjusted Sharpe Ratio
    - 不通过环境，而是顺序滚动窗口构造观测，使用确定性策略（均值动作）
    """
    df_test = pd.read_csv(test_csv_path)
    df_test = df_test.sort_values("date_id").reset_index(drop=True)

    # 用训练集统计量对 test 特征做相同预处理
    df_test_proc, _, _ = preprocess_features(df_test, feature_cols, mean=feat_mean, std=feat_std)

    n = len(df_test_proc)
    if n <= window_len + 1:
        print("test.csv 样本太少，无法进行有效评估。")
        return

    positions = []
    fr_list = []
    rf_list = []
    date_ids = []

    prev_pos = 0.0

    for idx in range(window_len, n):
        window = df_test_proc.loc[idx - window_len : idx - 1, feature_cols].values  # (window_len, D)
        obs = np.concatenate([window.flatten(), np.array([prev_pos], dtype=np.float32)], axis=0).astype(np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, log_std, val = ppo.ac(obs_t)
            raw = mean[0, 0]
            action = 2.0 * torch.sigmoid(raw)
        pos = float(action.cpu().item())

        row = df_test_proc.loc[idx]
        fwd = float(row["forward_returns"])
        rf = float(row["risk_free_rate"])
        d_id = row["date_id"]

        positions.append(pos)
        fr_list.append(fwd)
        rf_list.append(rf)
        date_ids.append(d_id)

        prev_pos = pos

    # 构造与竞赛一致的 score() 输入
    fr_arr = np.array(fr_list, dtype=np.float64)
    rf_arr = np.array(rf_list, dtype=np.float64)
    did_arr = np.array(date_ids)

    solution = pd.DataFrame(
        {
            "date_id": did_arr,
            "forward_returns": fr_arr,
            "risk_free_rate": rf_arr,
            "market_forward_excess_returns": fr_arr - rf_arr,
        }
    )
    submission = pd.DataFrame(
        {
            "date_id": did_arr,
            "allocation": np.array(positions, dtype=np.float64),
        }
    )

    test_score = score(solution, submission, row_id_column_name="date_id")
    print("\n" + "=" * 60)
    print(f"PPO 策略在 test.csv 上的最终 Adjusted Sharpe（score.py::score）: {test_score:.6f}")
    print("=" * 60 + "\n")

# -------------------------
# 如果作为脚本运行
# -------------------------
if __name__ == "__main__":
    train_example()
