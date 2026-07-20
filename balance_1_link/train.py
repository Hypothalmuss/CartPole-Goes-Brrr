import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import SinglePendulumCartEnv


TIMESTEPS = 200_000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENT_COEF = 0.03
VF_COEF = 0.5
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
MAX_EPISODE_STEPS = 500
N_EVAL_EPISODES = 5
CHECKPOINT_FREQ = 50_000

OUT_DIR = Path(__file__).parent / "output"


def make_env(render: bool = False):
    return SinglePendulumCartEnv(
        render_mode="human" if render else None,
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def train():
    env = DummyVecEnv([lambda: make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    os.makedirs(OUT_DIR, exist_ok=True)

    class ActionLogger(BaseCallback):
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
        save_path=str(OUT_DIR),
        name_prefix="ppo_single_pendulum",
    )

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
        tensorboard_log=str(OUT_DIR / "tensorboard"),
        device="auto",
    )

    print(f"Training for {TIMESTEPS} timesteps ...")
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[checkpoint_callback, ActionLogger()],
        progress_bar=True,
    )

    model.save(str(OUT_DIR / "ppo_single_pendulum_final.zip"))
    env.save(str(OUT_DIR / "vecnormalize.pkl"))

    print("Running evaluation episodes ...")
    rewards = []
    actions_log = []
    obs = env.reset()
    step_count = 0
    for _ in range(N_EVAL_EPISODES * MAX_EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward[0])
        actions_log.append(abs(action[0, 0]))
        step_count += 1
        if done[0] or step_count >= MAX_EPISODE_STEPS:
            obs = env.reset()
            step_count = 0
    print(f"Mean reward over evaluation: {np.mean(rewards):.3f}  +/- {np.std(rewards):.3f}")
    print(f"Mean |action|: {np.mean(actions_log):.4f}  +/- {np.std(actions_log):.4f}  "
          f"(max: {np.max(actions_log):.2f}, min: {np.min(actions_log):.4f})")

    env.close()


def evaluate():
    model_path = OUT_DIR / "ppo_single_pendulum_final.zip"
    if not model_path.exists():
        checkpoints = sorted(OUT_DIR.glob("ppo_single_pendulum_*_steps.zip"))
        if not checkpoints:
            print(f"No trained model found in {OUT_DIR}. Run `python train.py` first.")
            return
        model_path = checkpoints[-1]
        print(f"No final model found — loading latest checkpoint: {model_path.name}")

    env = make_env(render=True)
    model = PPO.load(str(model_path), env=env)

    print("Showing trained agent in PyBullet GUI — close the window to exit.")
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        evaluate()
