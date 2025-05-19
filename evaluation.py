"""
Assessing Provincial Carbon Efficiency (CE)

"""

import os
import argparse
import random
import warnings
from collections import deque
from sklearn.metrics import explained_variance_score, r2_score

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from xgboost import XGBRegressor

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

warnings.filterwarnings("ignore")

# ---------------- Global Configuration ---------------- #
SEED = 42
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_FILE = "data.xlsx"
XGB_FILE = "xgb_model_eval.pkl"
PPO_FILE = "ppo_policy_eval.zip"
BEST_PPO_DIR = "ppo_best"

FEATURES = ["AI", "ENE", "POP", "GDP", "IND", "GOV", "OPEN"]
TARGET = "CE"
TRAIN_YEARS = list(range(2013, 2021))
HIST_Y = list(range(2013, 2020))
PRED_YRS = 5


# ========== dataLoading ========== #
def load_data() -> pd.DataFrame:
    """Read Excel and rename the column, returning the sorted DataFrame"""
    df = pd.read_excel(DATA_FILE)
    rename = {
        "碳排放效率": "CE",
        "人工智能发展水平": "AI",
        "能源利用效率": "ENE",
        "人口规模": "POP",
        "经济发展水平": "GDP",
        "产业结构": "IND",
        "政府宏观调控水平": "GOV",
        "对外开放程度": "OPEN",
    }
    df = df.rename(columns=rename)
    df = df[["省份", "年份"] + FEATURES + [TARGET]].sort_values(["省份", "年份"])
    return df.reset_index(drop=True)
# ========== Train / Load XGBoost ========== #
def get_xgb(df: pd.DataFrame, *, skip_train: bool = False) -> XGBRegressor:
    """
    Returns the trained XGBRegressor
    - skip_train=True and the file is present -> Load directly
    - Otherwise, use TPOT to search for the optimal hyperparameters and save them
    """
    if skip_train and os.path.exists(XGB_FILE):
        print("XGBRegressor loaded")
        return joblib.load(XGB_FILE)

    # Split training set by year (2013–2020)
    train_df = df[df["年份"] <= 2020]
    X_train, y_train = train_df[FEATURES], train_df[TARGET]

    print("Searching for the optimal XGBRegressor with TPOT ...")
    tpot_cfg = {
        "xgboost.XGBRegressor": {
            "n_estimators": list(range(100, 301, 25)),
            "max_depth": list(range(3, 11)),
            "learning_rate": [round(x, 3) for x in np.linspace(0.01, 0.2, 20)],
            "subsample": [round(x, 2) for x in np.linspace(0.6, 1.0, 9)],
            "min_child_weight": list(range(1, 11)),
        }
    }
    tpot = TPOTRegressor(
        generations=20, population_size=60, config_dict=tpot_cfg,
        scoring="r2", random_state=SEED, verbosity=2, disable_update_check=True, template="XGBRegressor",
    )
    tpot.fit(X_train, y_train)
    best_xgb: XGBRegressor = tpot.fitted_pipeline_.steps[-1][1]
    joblib.dump(best_xgb, XGB_FILE)
    print("XGBRegressor saved to", XGB_FILE)
    y_pred_tr = best_xgb.predict(X_train)
    ev_tr = explained_variance_score(y_train, y_pred_tr)
    mae_tr = np.mean(np.abs(y_pred_tr - y_train))
    r2_tr  = best_xgb.score(X_train, y_train)
    print(f"[XGB Training Set] EV={ev_tr:.4f}, MAE={mae_tr:.4f}, R2={r2_tr:.4f}")
    return best_xgb


# ========== Adaptive Entropy Coefficient Callback ========== #
class AdaptiveEntropyCallback(BaseCallback):
    """Linearly decay ent_coef from initial to final value."""

    def __init__(self, initial=0.02, final=0.005, total_steps=3e5, verbose=0):
        super().__init__(verbose)
        self.initial, self.final, self.total = initial, final, total_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total)
        self.model.ent_coef = self.initial * (1 - progress) + self.final * progress
        return True


# ========== Calculate Annual Growth Rate by Province ========== #
def build_diff(df: pd.DataFrame) -> dict:
    diff = {}
    for province, grp in df.groupby("省份"):
        g = grp.set_index("年份").loc[TRAIN_YEARS]
        for y in HIST_Y:
            diff[(province, y)] = ((g.loc[y + 1, FEATURES] - g.loc[y, FEATURES]) / g.loc[y, FEATURES] * 100).values.astype(
                np.float32
            )
    return diff

# ========== Reward Scaling Wrapper ========== #
class RewardScalerWrapper(gym.Wrapper):
    """Logarithmic error + dynamic scaling for stable training"""

    def __init__(self, env, init_scale=20.0, adjust_every=50):
        super().__init__(env)
        self.scale = init_scale
        self.buffer = deque(maxlen=100)
        self.every, self.steps = adjust_every, 0

    def step(self, action):
        obs, r, d, tr, info = self.env.step(action)
        r = -np.log(abs(r) + 1e-4)  # Apply logarithmic transformation
        self.buffer.append(r)
        self.steps += 1
        r *= self.scale  # Apply scaling

        # Adaptively adjust scale every `adjust_every` steps
        if self.steps % self.every == 0 and self.buffer:
            m = abs(np.mean(self.buffer))
            if m < 0.3:
                self.scale *= 1.5
            elif m > 3.0:
                self.scale *= 0.7
            self.scale = np.clip(self.scale, 1.0, 500.0)
        return obs, r, d, tr, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ===== Build the wrapped environment =====
def build_wrapped_env(df, diff_dict, xgb_model):
    """Return a single-process environment: MaskablePolicyImpactEnv → RewardScaler → ActionMasker"""
    base_env = MaskablePolicyImpactEnv(df, diff_dict, xgb_model)  # Core env with action mask
    return ActionMasker(
        RewardScalerWrapper(base_env),  # Log-transformed and scaled rewards
        lambda e: e.get_action_mask()  # Mask function
    )

# ========== Custom Gym Environment ========== #
class PolicyImpactEnv(gym.Env):
    """
    Observation: 7-dimensional features + 1 normalized year value
    Action: Select a (province, year) pair whose historical growth rate is applied to the current province
    Reward: -abs(predicted CE - actual CE)
    """

    def __init__(self, df, diff_dict, xgb_model, penalty=5.0):
        super().__init__()
        self.df, self.diff, self.model, self.penalty = df, diff_dict, xgb_model, penalty

        self.cities = sorted(df["省份"].unique())
        self.pairs = [(c, y) for c in self.cities for y in HIST_Y]
        self.max_year = 2020

        self.action_space = Discrete(len(self.pairs))
        self.observation_space = Box(-np.inf, np.inf, (len(FEATURES) + 1,), np.float32)
        self._reset_episode()

    # ---------- Private Helper Methods ---------- #
    def _reset_episode(self):
        self.province = random.choice(self.cities)
        self.year = random.choice(HIST_Y)
        self.done = False

    def _state_vec(self):
        feats = self.df.query("省份==@self.province and 年份==@self.year").iloc[0][FEATURES].values.astype(np.float32)
        year_norm = (self.year - 2013) / 9.0
        return np.concatenate([feats, [year_norm]])

    # ---------- Gym Interface ---------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_episode()
        return self._state_vec(), {}

    def step(self, a_id):
        if self.done:
            raise RuntimeError("Episode has ended. Please call reset().")

        province_a, year_a = self.pairs[a_id]

        # Invalid action: mismatched province or future year
        if province_a != self.province or year_a > self.year:
            self.done = True
            return self._state_vec(), -self.penalty, self.done, False, {}

        diff_vec = self.diff[(province_a, year_a)]
        cur_feats = self.df.query("省份==@self.province and 年份==@self.year").iloc[0][FEATURES].values.astype(np.float32)
        next_feats = cur_feats + cur_feats * diff_vec / 100.0
        pred_ce = self.model.predict(next_feats.reshape(1, -1))[0]

        # Reward: negative absolute error
        if self.year + 1 > self.max_year:
            r = 0.0
            self.done = True
        else:
            real_ce = self.df.query("省份==@self.province and 年份==@self.year+1").iloc[0][TARGET]
            r = -abs(pred_ce - real_ce)

        self.year += 1
        if self.year >= self.max_year:
            self.done = True

        return self._state_vec(), r, self.done, False, {}


# ========== Action Mask Support ========== #
class MaskablePolicyImpactEnv(PolicyImpactEnv):
    def get_action_mask(self):
        mask = np.zeros(len(self.pairs), dtype=np.int8)
        mask[[i for i, (c, y) in enumerate(self.pairs) if c == self.province and y <= self.year]] = 1
        return mask


# ========== Train PPO ========== #
def train_ppo(df, diff, xgb):
    def mask_fn(env):
        return env.get_action_mask()

    vec_env = make_vec_env(
        lambda: ActionMasker(RewardScalerWrapper(MaskablePolicyImpactEnv(df[df["年份"] <= 2020], diff, xgb)), mask_fn),
        n_envs=1,
        seed=SEED,
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_log",
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        ent_coef=0.02,
        clip_range=0.3,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    eval_env = make_vec_env(lambda: RewardScalerWrapper(PolicyImpactEnv(df, diff, xgb)), n_envs=1, seed=SEED + 1)
    eval_cb = EvalCallback(eval_env, best_model_save_path=BEST_PPO_DIR, eval_freq=10_000, deterministic=True, verbose=0)

    model.learn(total_timesteps=50_000, callback=[AdaptiveEntropyCallback(), eval_cb])
    model.save(PPO_FILE)
    print("PPO model saved to", PPO_FILE)
    return model


# ========== Evaluate 2021‑2022 ========== #
def evaluate(df, diff, ppo, xgb):
    env = ppo.get_env().envs[0].unwrapped
    rec = []

    for province in sorted(df["省份"].unique()):
        feats = df.query("省份==@province and 年份==2020").iloc[0][FEATURES].values.astype(np.float32)
        year = 2020

        for _ in range(2):
            state = np.concatenate([feats, [(year - 2013) / 9]]).reshape(1, -1)

            env.province, env.year = province, year
            action, _ = ppo.predict(state, deterministic=True, action_masks=env.get_action_mask())
            province_a, year_a = env.pairs[int(action)]

            diff_vec = diff[(province_a, year_a)]
            feats = feats + feats * diff_vec / 100.0
            pred_ce = xgb.predict(feats.reshape(1, -1))[0]

            real_ce = df.query("省份==@province and 年份==@year+1").iloc[0][TARGET]
            rec.append(
                dict(
                    province=province,
                    year=year + 1,
                    pred_CE=pred_ce,
                    real_CE=real_ce,
                    abs_error=abs(pred_ce - real_ce),
                    action_from=f"{province_a}-{year_a}",
                )
            )
            year += 1

    pd.DataFrame(rec).to_csv("eval_2021_2022.csv", index=False)
    print("Evaluation results saved to eval_2021_2022.csv")
    return rec


def analyze_overfit(records):
    """Output MAE, max error, and explained variance (EV)"""
    y_true = [r["real_CE"] for r in records]
    y_pred = [r["pred_CE"] for r in records]
    abs_errors = [abs(p - t) for p, t in zip(y_pred, y_true)]

    mae = np.mean(abs_errors)
    mx = np.max(abs_errors)
    ev = explained_variance_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n[Evaluation Set: 2021–2022]")
    print(f"Explained Variance (EV): {ev:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Max Absolute Error: {mx:.4f}")
    print(f"R^2 Score: {r2:.4f}")


# ========== Main Entry ========== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true", help="Only load models and run evaluation")
    args = parser.parse_args()

    df = load_data()

    # Split training and evaluation sets: only use data up to 2020 for training, full data for evaluation
    train_df = df[df["年份"] <= 2020]
    diff = build_diff(train_df)
    xgb = get_xgb(train_df, skip_train=args.infer)

    if args.infer and os.path.exists(PPO_FILE):
        print("Loading PPO model ...")
        env = make_vec_env(lambda: build_wrapped_env(train_df, diff, xgb),
                           n_envs=1, seed=SEED)
        ppo = MaskablePPO.load(PPO_FILE, env=env, device="cpu")
    else:
        ppo = train_ppo(train_df, diff, xgb)

    # Use full dataset for evaluation (predict 2021–2022)
    records = evaluate(df, diff, ppo, xgb)
    analyze_overfit(records)


if __name__ == "__main__":
    main()
