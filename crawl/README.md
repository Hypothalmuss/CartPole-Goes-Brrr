# Crawl-Pole

Custom Gymnasium environment: same 2-link pendulum, but the cart is a free body
on the ground plane and the pole uses joint torque to crawl forward.

## Setup

```bash
pip install pybullet gymnasium stable-baselines3 torch numpy
```

## Usage

```bash
python train_crawl.py                    # train from scratch
python train_crawl.py --timesteps 5000   # smoke test
python train_crawl.py --mode eval        # watch trained agent
python eval_crawl.py                     # GUI playback
python eval_crawl.py --diagnose          # headless distance stats
```

Training logs: `output/tensorboard/` — view with `tensorboard --logdir output/tensorboard`.

## Spec

| Property | Value |
|----------|-------|
| URDF | `cartpole_crawl.urdf` |
| Cart | free body (no rail), `useFixedBase=False` |
| Links | hip: 1.0 m / 0.5 kg, elbow: 0.7 m / 0.3 kg |
| Action | continuous torque [hip, elbow], [-20, 20] / [-15, 15] Nm |
| Observation | 8 values: cart x/vx/z, hip angle/vel, elbow angle/vel, tip_contact |
| Reward | forward progress (grounded only) - energy - height + tip contact |
| Termination | cart_z > 0.15 m (flew) or truncation at 1000 steps |

## Key Differences from Balance

- No rail joint — cart slides freely on ground plane via contact friction
- Hip/elbow joints are torque-actuated (`TORQUE_CONTROL`), not passive
- Hip origin pitched 90° so links lie flat at `hip_angle=0`
- Progress only counts when cart is grounded (prevents jump exploit)

## Key Files

- `env_crawl.py` — `CrawlPoleEnv(gym.Env)`
- `train_crawl.py` — PPO training with VecNormalize + checkpointing
- `eval_crawl.py` — GUI playback and diagnostic crawl-distance analysis
- `cartpole_crawl.urdf` — URDF with free-base cart + actuated revolute joints
