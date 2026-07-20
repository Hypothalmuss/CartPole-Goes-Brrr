"""
Load a trained PPO model and watch it balance the double pendulum in PyBullet GUI.

Usage:
    python eval.py                              # GUI playback (infinite)
    python eval.py --diagnose                   # headless termination-reason breakdown
    python eval.py --checkpoint path/to/model   # use a specific checkpoint
"""

import argparse
from collections import Counter, deque
from pathlib import Path

import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import DoublePendulumCartEnv


MAX_EVAL_EPISODES = 20


def main():
    parser = argparse.ArgumentParser(description="Visualise a trained PPO agent on the double pendulum.")
    parser.add_argument(
        "--checkpoint", "-c", type=str, default=None,
        help="Path to a specific model checkpoint. Defaults to the final model.",
    )
    parser.add_argument(
        "--diagnose", "-d", action="store_true",
        help="Run headless termination-reason breakdown instead of GUI playback.",
    )
    args = parser.parse_args()

    out = Path(__file__).parent / "output"

    if args.checkpoint is not None:
        model_path = Path(args.checkpoint)
    else:
        model_path = out / "ppo_double_pendulum_final.zip"
        if not model_path.exists():
            checkpoints = sorted(out.glob("ppo_double_pendulum_*_steps.zip"))
            if not checkpoints:
                print(f"No trained model found in {out}. Run `python train.py` first.")
                return
            model_path = checkpoints[-1]
            print(f"No final model — using latest checkpoint: {model_path.name}")

    if args.diagnose:
        diagnose(model_path)
    else:
        playback(model_path)


def _make_normalized_env(render_mode: str | None, model_path: Path):
    """Build the raw env, wrap in DummyVecEnv + VecNormalize, load saved stats."""
    raw = DoublePendulumCartEnv(render_mode=render_mode, max_episode_steps=500)
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

    print("Showing trained agent — close the PyBullet GUI window to exit.")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done[0]:
            obs = env.reset()


def diagnose(model_path: Path):
    env, raw = _make_normalized_env(None, model_path)
    model = PPO.load(str(model_path), env=env)

    term_counts = Counter()
    episode_count = 0
    recent_steps = deque(maxlen=10)
    hip_angle_termination_samples = []
    elbow_angle_termination_samples = []
    print(f"Running {MAX_EVAL_EPISODES} diagnostic episodes ...")

    obs = env.reset()
    while episode_count < MAX_EVAL_EPISODES:
        action, _ = model.predict(obs, deterministic=True)

        # Query ground-truth velocities directly from PyBullet, bypassing any
        # stacked observation normalization (VecNormalize on top of manual scale).
        hip_state = p.getJointState(raw.cart_body, raw.joint_hip)
        elbow_state = p.getJointState(raw.cart_body, raw.joint_elbow)
        hip_vel_raw = hip_state[1]
        elbow_vel_raw = elbow_state[1]

        obs, reward, done, info = env.step(action)
        recent_steps.append((float(action[0, 0]), float(hip_vel_raw), float(elbow_vel_raw)))

        if done[0]:
            reason = info[0].get("term_reason", "truncated")
            term_counts[reason] += 1
            if reason == "hip_angle":
                hip_angle_termination_samples.append(list(recent_steps))
            elif reason == "elbow_angle":
                elbow_angle_termination_samples.append(list(recent_steps))
            episode_count += 1
            obs = env.reset()
            recent_steps.clear()

    print(f"\n--- Termination reason breakdown over {MAX_EVAL_EPISODES} episodes ---")
    for reason, count in term_counts.most_common():
        print(f"  {reason}: {count} ({100 * count // MAX_EVAL_EPISODES}%)")

    print(f"\n--- Last 10 (action, hip_vel, elbow_vel) before hip_angle terminations (first 3) ---")
    for i, sample in enumerate(hip_angle_termination_samples[:3]):
        print(f"Episode {i}:")
        for step_i, (a, hv, ev) in enumerate(sample):
            print(f"  t-{len(sample)-step_i}: action={a:+.2f} (max={raw.max_force}), "
                  f"hip_vel={hv:+.2f}, elbow_vel={ev:+.2f}")

    print(f"\n--- Last 10 (action, hip_vel, elbow_vel) before elbow_angle terminations (first 3) ---")
    for i, sample in enumerate(elbow_angle_termination_samples[:3]):
        print(f"Episode {i}:")
        for step_i, (a, hv, ev) in enumerate(sample):
            print(f"  t-{len(sample)-step_i}: action={a:+.2f} (max={raw.max_force}), "
                  f"hip_vel={hv:+.2f}, elbow_vel={ev:+.2f}")
    env.close()


if __name__ == "__main__":
    main()
