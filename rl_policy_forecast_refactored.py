"""

Forecasting province Carbon Emission Efficiency (CE)

"""

# ======== Core Libraries ========
import os
import argparse
import random
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch

# ======== Stable Baselines3 (SB3) ========
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

# ======== Supervised Models ========
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from xgboost import XGBRegressor

# ======== Gym Environment ========
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

# ======== Miscellaneous ========
import warnings

warnings.filterwarnings("ignore")

# ---------------- Set Random Seed ---------------- #
SEED = 42
set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- Constants and Configuration ---------------- #
DATA_FILE = "data.xlsx"
XGB_FILE = "xgb_model.pkl"           # Trained XGBRegressor model
PPO_FILE = "ppo_policy.zip"          # Trained PPO policy
BEST_PPO_DIR = "ppo_best"            # Directory for EvalCallback to save the best model

FEATURES = ["AI", "ENE", "POP", "GDP", "IND", "GOV", "OPEN"]
TARGET = "CE"
YEARS = list(range(2013, 2023))              # Years from 2013 to 2022
HISTORY_YEARS = list(range(2013, 2022))      # Years used to compute growth differentials
PRED_YEARS = 5                               # Number of future years to predict

# ============== Data Loading and Preprocessing ============== #
def load_data() -> pd.DataFrame:
    """Read Excel data and standardize column names"""
    df = pd.read_excel(DATA_FILE)
    rename_map = {
        "碳排放效率": "CE",
        "人工智能发展水平": "AI",
        "能源利用效率": "ENE",
        "人口规模": "POP",
        "经济发展水平": "GDP",
        "产业结构": "IND",
        "政府宏观调控水平": "GOV",
        "对外开放程度": "OPEN",
    }
    df = df.rename(columns=rename_map)
    df = df[["省份", "年份"] + FEATURES + [TARGET]].sort_values(["省份", "年份"])
    return df.reset_index(drop=True)


# ============== Train or Load XGBoost ============== #
def get_xgb(df: pd.DataFrame, skip_train: bool = False) -> XGBRegressor:
    """
    Train or load an XGBRegressor depending on `skip_train` flag:
    - If skip_train is True and model file exists → Load the model
    - Otherwise → Use TPOT to search for best hyperparameters and save the model
    """
    if skip_train and os.path.exists(XGB_FILE):
        print("XGBRegressor loaded")
        return joblib.load(XGB_FILE)

    X, y = df[FEATURES], df[TARGET]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("Searching for optimal XGBRegressor via TPOT ...")
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
    tpot.fit(X_tr, y_tr)
    best_xgb: XGBRegressor = tpot.fitted_pipeline_.steps[-1][1]
    joblib.dump(best_xgb, XGB_FILE)
    print("XGBRegressor saved to", XGB_FILE)
    return best_xgb


# ================= Custom Callback: Adaptive Entropy Coefficient ================= #
class AdaptiveEntropyCallback(BaseCallback):
    """
    Linearly decays the entropy coefficient (ent_coef) during training.
    Encourages exploration early and exploitation later.
    """

    def __init__(self, initial=0.02, final=0.005, total_steps=3e5, verbose=0):
        super().__init__(verbose)
        self.initial = initial
        self.final = final
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_steps)
        self.model.ent_coef = self.initial * (1 - progress) + self.final * progress
        return True

# ============== Construct diff Dictionary ============== #
def build_diff(df: pd.DataFrame) -> dict:
    """
    Compute year-on-year percentage change of features for each province:
    diff_dict[(province, yr)] -> 7-dimensional percentage growth (%)
    """
    diff_dict = {}
    for province, grp in df.groupby("省份"):
        grp = grp.set_index("年份").loc[YEARS]
        for yr in HISTORY_YEARS:
            y0, y1 = grp.loc[yr, FEATURES], grp.loc[yr + 1, FEATURES]
            diff_dict[(province, yr)] = ((y1 - y0) / y0 * 100).values.astype(np.float32)
    return diff_dict


# ============== Reward Scaling Wrapper ============== #
class RewardScalerWrapper(gym.Wrapper):
    """
    Transforms the reward (negative absolute error):
    - Apply logarithmic transformation: small error → reward near 0; large error → more negative
    - Then apply dynamic scaling to amplify or attenuate rewards during training
    """

    def __init__(self, env, init_scale=20.0, adjust_every=50):
        super().__init__(env)
        self.scale = init_scale
        self.buffer = deque(maxlen=100)
        self.adjust_every = adjust_every
        self.steps = 0

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        reward = -np.log(abs(reward) + 1e-4)  # log-scaled error
        self.buffer.append(reward)
        self.steps += 1
        reward *= self.scale  # apply scaling

        # Adjust scale dynamically every N steps
        if self.steps % self.adjust_every == 0 and self.buffer:
            abs_mean = abs(np.mean(self.buffer))
            if abs_mean < 0.3:
                self.scale *= 1.5
            elif abs_mean > 3.0:
                self.scale *= 0.7
            self.scale = np.clip(self.scale, 1.0, 500.0)
        return obs, reward, done, trunc, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# ============== Custom Gym Environment ============== #
class PolicyImpactEnv(gym.Env):
    """
    - State: 7 economic/energy indicators + normalized year
    - Action: Select a historical growth pattern (province, year) and apply it to the current province
    - Reward: -abs(predicted CE - actual CE)
    """

    def __init__(self, df, diff_dict, xgb_model, penalty=5.0):
        super().__init__()
        self.df = df
        self.diff_dict = diff_dict
        self.model = xgb_model
        self.penalty = penalty

        self.cities = sorted(df["省份"].unique())
        # All possible actions (only valid if same province and year ≤ current year)
        self.action_pairs = [(c, y) for c in self.cities for y in HISTORY_YEARS]

        self.action_space = Discrete(len(self.action_pairs))
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(len(FEATURES) + 1,), dtype=np.float32
        )
        self._reset_episode()

    # ---------- Internal Helpers ---------- #
    def _reset_episode(self):
        self.province = random.choice(self.cities)
        self.year = random.choice(HISTORY_YEARS)
        self.done = False

    def _state_vec(self):
        """Return current state vector: features + normalized year"""
        row = self.df.query("省份==@self.province and 年份==@self.year")
        feats = row.iloc[0][FEATURES].values.astype(np.float32)
        year_norm = (self.year - 2013) / 9.0
        return np.concatenate([feats, [year_norm]])

    # ---------- Gym API ---------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_episode()
        return self._state_vec(), {}

    def step(self, action_id):
        if self.done:
            raise RuntimeError("Episode has ended")

        province_a, year_a = self.action_pairs[action_id]
        # Invalid action: mismatched province or future year
        if province_a != self.province or year_a > self.year:
            reward = -self.penalty
            self.done = True
            return self._state_vec(), reward, self.done, False, {}

        diff = self.diff_dict[(province_a, year_a)]
        cur_feats = self.df.query("省份==@self.province and 年份==@self.year").iloc[0][FEATURES].values.astype(np.float32)
        next_feats = cur_feats + cur_feats * diff / 100.0  # Apply percentage change
        pred_ce = self.model.predict(next_feats.reshape(1, -1))[0]

        # If reached final year: no reward, just terminate
        if self.year + 1 > YEARS[-1]:
            reward = 0.0
            self.done = True
        else:
            real_ce = self.df.query("省份==@self.province and 年份==@self.year+1").iloc[0][TARGET]
            reward = -abs(pred_ce - real_ce)  # Negative absolute error

        self.year += 1
        if self.year >= YEARS[-1]:
            self.done = True

        return self._state_vec(), reward, self.done, False, {}

# -------- Environment with Action Masking (Prevent Invalid Actions) -------- #
class MaskablePolicyImpactEnv(PolicyImpactEnv):
    def get_action_mask(self):
        """Return a 0/1 mask indicating valid actions"""
        mask = np.zeros(len(self.action_pairs), dtype=np.int8)
        legal = [(c == self.province and y <= self.year) for c, y in self.action_pairs]
        mask[np.where(legal)] = 1
        return mask

# ============== Build the Training Environment ============== #
def build_wrapped_env(df, diff_dict, xgb_model):
    mask_fn = lambda e: e.get_action_mask()
    base_env = MaskablePolicyImpactEnv(df, diff_dict, xgb_model)
    return ActionMasker(RewardScalerWrapper(base_env), mask_fn)
# ============== Train PPO ============== #
def train_ppo(df, diff_dict, xgb_model) -> MaskablePPO:
    """Builds a vectorized environment, trains MaskablePPO, and saves the model"""

    def mask_fn(env):  # For use with ActionMasker
        return env.get_action_mask()

    # Training environment
    vec_env = make_vec_env(
        lambda: build_wrapped_env(df, diff_dict, xgb_model),
        n_envs=1,
        seed=SEED
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
        device="cpu"
    )

    # Evaluation environment
    eval_env = make_vec_env(
        lambda: RewardScalerWrapper(PolicyImpactEnv(df, diff_dict, xgb_model)),
        n_envs=1,
        seed=SEED + 1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_PPO_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0
    )

    # Start training
    model.learn(
        total_timesteps=50_000,  # Increase if longer training is needed
        callback=[AdaptiveEntropyCallback(), eval_cb]
    )

    model.save(PPO_FILE)
    print("PPO model saved to:", PPO_FILE)
    return model

# ============== Forecast CE for the Next 5 Years ============== #
def forecast(df, diff_dict, ppo, xgb):
    """
    Use trained MaskablePPO + XGB to forecast CE for 2023–2027.
    - Start from 2022 indicators for each province
    - For each year: apply optimal action → update features → predict next year's CE using XGB
    - If real value exists, compute abs_error (optional)
    """
    env_unwrapped = ppo.get_env().envs[0].unwrapped
    rec = []

    for province in sorted(df["省份"].unique()):
        feats = df.query("省份==@province and 年份==2022").iloc[0][FEATURES].values.astype(np.float32)
        year = 2022

        for _ in range(PRED_YEARS):
            state = np.concatenate([feats, [(year - 2013) / 9]]).reshape(1, -1)

            # Update env's internal state for masking
            env_unwrapped.province = province
            env_unwrapped.year = year
            mask = env_unwrapped.get_action_mask()

            # Predict a valid action
            action, _ = ppo.predict(state, deterministic=True, action_masks=mask)
            province_a, year_a = env_unwrapped.action_pairs[int(action)]

            # Apply feature change
            diff = diff_dict[(province_a, year_a)]
            feats = feats + feats * diff / 100.0
            pred_ce = xgb.predict(feats.reshape(1, -1))[0]


            rec.append({
                "province": province,
                "year": year + 1,
                "pred_CE": pred_ce,
                "action_from": f"{province_a}-{year_a}",
            })
            year += 1

    out = pd.DataFrame(rec)
    out.to_excel("future_5yr_ce.xlsx", index=False)
    print("Forecast results for the next 5 years saved to future_5yr_ce.xlsx")
    return out


# ============== Main Entry Point ============== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true", help="Inference only: skip training")
    args = parser.parse_args()

    df = load_data()
    diff_dict = build_diff(df)
    xgb = get_xgb(df, skip_train=args.infer)

    # Train or load PPO
    if args.infer and os.path.exists(PPO_FILE):
        print("Loading PPO model ...")
        env = build_wrapped_env(df, diff_dict, xgb)
        ppo = MaskablePPO.load(PPO_FILE, env=env, device="cpu")
    else:
        ppo = train_ppo(df, diff_dict, xgb)

    forecast(df, diff_dict, ppo, xgb)


if __name__ == "__main__":
    main()
