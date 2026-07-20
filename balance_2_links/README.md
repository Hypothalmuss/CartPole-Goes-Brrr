# Balance-Pole

Custom Gymnasium environment: asymmetric double-pendulum on a rail-mounted cart,
trained with SB3 PPO.

## Setup

```bash
pip install pybullet gymnasium stable-baselines3 torch numpy
```

## Usage

```bash
python train.py                          # train from scratch
python train.py --mode eval              # watch trained agent
python eval.py                           # GUI playback
python eval.py --diagnose                # headless termination breakdown
```

Training logs: `output/tensorboard/` — view with `tensorboard --logdir output/tensorboard`.

## Spec

| Property | Value |
|----------|-------|
| URDF | `cartpole_double.urdf` |
| Links | hip: 1.0 m / 0.5 kg, elbow: 0.7 m / 0.3 kg |
| Action | continuous force (N) on cart, [-20, 20] |
| Observation | 6 values: cart pos/vel, hip angle/vel, elbow angle/vel |
| Reward | weighted uprightness (cos) + survival - penalties |
| Reset | random perturbation around upright equilibrium |

## Key Files

- `env.py` — `DoublePendulumCartEnv(gym.Env)`
- `train.py` — PPO training with VecNormalize + checkpointing
- `eval.py` — GUI playback and diagnostic termination analysis
- `cartpole_double.urdf` — URDF with prismatic rail joint + passive revolute joints
