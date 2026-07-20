"""
Load a trained PPO model and watch it crawl in PyBullet GUI.

Usage:
    python eval_crawl.py                              # GUI playback (infinite)
    python eval_crawl.py --diagnose                    # headless progress breakdown
    python eval_crawl.py --checkpoint path/to/model    # use a specific checkpoint
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_crawl import CrawlPoleEnv


MAX_EVAL_EPISODES = 20


def main():
    parser = argparse.ArgumentParser(description="Visualise a trained PPO agent on the crawl task.")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to a specific model checkpoint. Defaults to the final model.",
    )
    parser.add_argument(
        "--diagnose", "-d", action="store_true",
        help="Run headless progress-distance breakdown instead of GUI playback.",
    )
    args = parser.parse_args()

    out = Path(__file__).parent / "output"

    if args.checkpoint is not None:
        model_path = Path(args.checkpoint)
    else:
        model_path = out / "ppo_crawl_pole_final.zip"
        if not model_path.exists():
            checkpoints = sorted(out.glob("ppo_crawl_pole_*_steps.zip"))
            if not checkpoints:
                print(f"No trained model found in {out}. Run `python train_crawl.py` first.")
                return
            model_path = checkpoints[-1]
            print(f"No final model -- using latest checkpoint: {model_path.name}")

    if args.diagnose:
        diagnose(model_path)
    else:
        playback(model_path)


def _make_normalized_env(render_mode: str | None, model_path: Path):
    """Build the raw env, wrap in DummyVecEnv + VecNormalize, load saved stats."""
    raw = CrawlPoleEnv(render_mode=render_mode, max_episode_steps=1000)
    env = DummyVecEnv([lambda: raw])
    vecnorm_path = model_path.parent / "vecnormalize.pkl"
    if not vecnorm_path.exists():
        print(f"ERROR: {vecnorm_path} not found. Cannot run eval without VecNormalize stats.")
        raise SystemExit(1)
    env = VecNormalize.load(str(vecnorm_path), env)
    env.training = False
    env.norm_reward = False
    return env, raw


def playback(model_path: Path):
    env, _ = _make_normalized_env("human", model_path)
    model = PPO.load(str(model_path), env=env)

    print("Showing trained agent -- close the PyBullet GUI window to exit.")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done[0]:
            obs = env.reset()


def diagnose(model_path: Path):
    env, raw = _make_normalized_env(None, model_path)
    model = PPO.load(str(model_path), env=env)

    episode_count = 0
    distances = []
    term_reasons = []
    start_x = None

    print(f"Running {MAX_EVAL_EPISODES} diagnostic episodes ...")

    obs = env.reset()
    start_x = raw.prev_cart_x
    while episode_count < MAX_EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            distance = info[0].get("cart_x", raw.prev_cart_x) - start_x
            distances.append(distance)
            term_reasons.append(info[0].get("term_reason", "truncated"))
            episode_count += 1
            obs = env.reset()
            start_x = raw.prev_cart_x

    distances = np.array(distances)
    print(f"\n--- Crawl distance over {MAX_EVAL_EPISODES} episodes ---")
    print(f"  mean:   {distances.mean():+.3f} m")
    print(f"  std:    {distances.std():.3f} m")
    print(f"  best:   {distances.max():+.3f} m")
    print(f"  worst:  {distances.min():+.3f} m")

    from collections import Counter
    reason_counts = Counter(term_reasons)
    print(f"\n--- Termination reason breakdown ---")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count} ({100 * count // MAX_EVAL_EPISODES}%)")

    env.close()


if __name__ == "__main__":
    main()
