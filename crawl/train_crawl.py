"""
Train a PPO agent on the custom crawl-pole environment.

Usage:
    python train_crawl.py                          # train from scratch
    python train_crawl.py --timesteps 5000          # short smoke test
    python train_crawl.py --mode eval                # load saved model & watch it
"""

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_crawl import CrawlPoleEnv

# ------------------------------------------------------------------ #
#  Hyperparameters -- same starting point as the balance task, tuned #
#  for a longer horizon since crawling is a harder credit-assignment #
#  problem (reward is sparse-ish until a gait actually emerges).     #
# ------------------------------------------------------------------ #
TIMESTEPS = 1_000_000         # total timesteps -- crawling needs more exploration
LEARNING_RATE = 3e-4         # Adam lr
GAMMA = 0.99                 # discount factor
GAE_LAMBDA = 0.95            # GAE smoothing
CLIP_EPSILON = 0.2           # PPO clip range
ENT_COEF = 0.05              # higher exploration -- agent needs to discover ground contact
VF_COEF = 0.5                # value function loss coefficient
N_STEPS = 2048                # steps per update (rollout buffer size)
BATCH_SIZE = 64               # minibatch size
N_EPOCHS = 10                 # number of epochs per update
MAX_EPISODE_STEPS = 1000      # episode length
N_EVAL_EPISODES = 5           # episodes for eval reward tracking

# Where to save models & logs
OUT_DIR = Path(__file__).parent / "output"
CHECKPOINT_FREQ = 50_000     # save a checkpoint every N timesteps


def make_env(render: bool = False):
    return CrawlPoleEnv(
        render_mode="human" if render else None,
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def train(total_timesteps: int = TIMESTEPS):
    env = DummyVecEnv([lambda: make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # ---- Callbacks ----
    os.makedirs(OUT_DIR, exist_ok=True)

    class ActionLogger(BaseCallback):
        """Log mean |action| to detect collapsed action distribution."""
        def __init__(self):
            super().__init__()
            self.actions = []
        def _on_step(self) -> bool:
            if "actions" in self.locals:
                self.actions.extend(abs(self.locals["actions"]).flatten().tolist())
            return True
        def _on_rollout_end(self) -> None:
            if self.actions:
                mean_act = np.mean(self.actions)
                self.logger.record("train/mean_abs_action", mean_act)
                self.actions.clear()

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_STEPS, 1),
        save_path=OUT_DIR,
        name_prefix="ppo_crawl_pole",
    )

    # ---- PPO ----
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_EPSILON,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        verbose=1,
        tensorboard_log=OUT_DIR / "tensorboard",
        device="auto",  # auto-detects CUDA
    )

    print(f"Training for {total_timesteps} timesteps ...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, ActionLogger()],
        progress_bar=True,
    )

    final_path = OUT_DIR / "ppo_crawl_pole_final.zip"
    model.save(final_path)
    print(f"Model saved to {final_path}")
    vecnorm_path = OUT_DIR / "vecnormalize.pkl"
    env.save(str(vecnorm_path))
    print(f"VecNormalize stats saved to {vecnorm_path}")

    # Quick eval with action + progress stats
    print("Running evaluation episodes ...")
    rewards = []
    actions_log = []
    obs = env.reset()
    step_count = 0
    for _ in range(N_EVAL_EPISODES * MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
        actions_log.append(np.abs(action[0]).mean())
        step_count += 1
        if done[0] or step_count >= MAX_EPISODE_STEPS:
            obs = env.reset()
            step_count = 0
    print(f"Mean reward over evaluation: {np.mean(rewards):.3f}  +/- {np.std(rewards):.3f}")
    print(f"Mean |action|: {np.mean(actions_log):.4f}  +/- {np.std(actions_log):.4f}  "
          f"(max: {np.max(actions_log):.2f}, min: {np.min(actions_log):.4f})")

    env.close()


def evaluate():
    model_path = OUT_DIR / "ppo_crawl_pole_final.zip"
    if not model_path.exists():
        # try checkpoint
        checkpoints = sorted(OUT_DIR.glob("ppo_crawl_pole_*_steps.zip"))
        if not checkpoints:
            print(f"No trained model found in {OUT_DIR}. Run `python train_crawl.py` first.")
            return
        model_path = checkpoints[-1]
        print(f"No final model found -- loading latest checkpoint: {model_path.name}")

    raw_env = make_env(render=True)
    env = DummyVecEnv([lambda: raw_env])
    vecnorm_path = OUT_DIR / "vecnormalize.pkl"
    if vecnorm_path.exists():
        env = VecNormalize.load(str(vecnorm_path), env)
        env.training = False
        env.norm_reward = False
    model = PPO.load(model_path, env=env)

    print("Showing trained agent in PyBullet GUI -- close the window to exit.")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done[0]:
            obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override TIMESTEPS (useful for a short smoke test, e.g. --timesteps 5000).",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps or TIMESTEPS)
    else:
        evaluate()
