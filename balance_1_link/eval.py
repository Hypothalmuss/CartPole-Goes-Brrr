import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import SinglePendulumCartEnv


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def make_env(render_mode: str | None = None):
    def _init():
        env = SinglePendulumCartEnv(render_mode=render_mode, max_episode_steps=500)
        return env
    return _init


def playback():
    model = PPO.load(os.path.join(OUTPUT_DIR, "ppo_single_pendulum_final.zip"))

    env = DummyVecEnv([make_env(render_mode="human")])
    env = VecNormalize.load(os.path.join(OUTPUT_DIR, "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


def diagnose():
    model = PPO.load(os.path.join(OUTPUT_DIR, "ppo_single_pendulum_final.zip"))

    env = DummyVecEnv([make_env()])
    env = VecNormalize.load(os.path.join(OUTPUT_DIR, "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    n_episodes = 20
    term_counts = {"rail": 0, "pole_angle": 0, "truncated": 0}
    failure_log = {"rail": [], "pole_angle": [], "truncated": []}

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        step_buffer = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                cart_pos = float(obs[0, 0]) * 2.4
                pole_angle = float(obs[0, 2]) * 3.14159
                reason = info[0].get("term_reason", "truncated") if isinstance(info, list) else "truncated"
                if reason not in term_counts:
                    reason = "truncated"
                term_counts[reason] += 1
                last_steps = step_buffer[-10:]
                failure_log[reason].append({
                    "episode": ep,
                    "term_reason": reason,
                    "last_10_steps": last_steps,
                })

            raw_obs = obs[0]
            step_buffer.append({
                "action": float(action[0][0]),
                "cart_pos": float(raw_obs[0]),
                "cart_vel": float(raw_obs[1]),
                "pole_angle": float(raw_obs[2]),
                "pole_vel": float(raw_obs[3]),
            })

    print(f"\nTermination breakdown ({n_episodes} episodes):")
    for reason, count in sorted(term_counts.items()):
        print(f"  {reason}: {count}")

    print("\nLast 10 steps before each termination type:")
    for reason, episodes in failure_log.items():
        if not episodes:
            continue
        ep = episodes[-1]
        print(f"\n--- {reason} (episode {ep['episode']}) ---")
        for s in ep["last_10_steps"]:
            print(f"  a={s['action']:+7.3f}  "
                  f"px={s['cart_pos']:+7.3f}  "
                  f"vx={s['cart_vel']:+7.3f}  "
                  f"th={s['pole_angle']:+7.3f}  "
                  f"dth={s['pole_vel']:+7.3f}")

    env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        diagnose()
    else:
        playback()
