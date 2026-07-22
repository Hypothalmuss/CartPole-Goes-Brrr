import math
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


URDF_PATH = os.path.join(os.path.dirname(__file__), "cartpole_single.urdf")


class SinglePendulumCartEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.max_force = 25.0
        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        # Reward weights
        self.R_UPRIGHT = 1.0
        self.R_SURVIVAL = 0.1
        self.R_CART_POS = -0.5
        self.R_CART_VEL = -0.005
        self.R_POLE_VEL = -0.05
        self.R_CONTROL = -0.01
        self.R_JERK = -0.005

        # Limits
        self.rail_limit = 2.4
        self.angle_limit = 1.5

        # Normalisation scales
        self.pos_scale = self.rail_limit
        self.vel_scale = 10.0
        self.angle_scale = math.pi
        self.angvel_scale = 10.0

        self.physics_client: int | None = None
        self.cart_body: int | None = None

        self.joint_rail = 0
        self.joint_hip = 1

        self.last_action = 0.0

        self._setup_sim()

    def _setup_sim(self) -> None:
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf", [0, 0, 0])

        self.cart_body = p.loadURDF(URDF_PATH, [0, 0, 0])

        p.setJointMotorControl2(self.cart_body, self.joint_hip, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cart_body, self.joint_rail, p.VELOCITY_CONTROL, force=0)

    CURRICULUM_FALLING_START_PROB = 0.3

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_action = 0.0

        rail_pos = self.np_random.uniform(-0.1, 0.1)
        rail_vel = self.np_random.uniform(-0.5, 0.5)
        pole_angle_init = self.np_random.uniform(-0.2, 0.2)
        pole_vel_init = self.np_random.uniform(-0.3, 0.3)

        if self.np_random.random() < self.CURRICULUM_FALLING_START_PROB:
            pole_vel_init = self.np_random.uniform(-4.0, 4.0)

        p.resetJointState(self.cart_body, self.joint_rail, targetValue=rail_pos, targetVelocity=rail_vel)
        p.resetJointState(self.cart_body, self.joint_hip, targetValue=pole_angle_init, targetVelocity=pole_vel_init)

        p.stepSimulation()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_force, self.max_force)[0]

        p.setJointMotorControl2(
            self.cart_body, self.joint_rail, p.TORQUE_CONTROL,
            force=float(action),
        )
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(1.0 / 240.0)
        self.step_count += 1

        obs = self._get_obs()
        cart_pos, cart_vel, pole_angle, pole_vel = obs

        uprightness = math.cos(pole_angle * self.angle_scale)

        penalty = (
            abs(cart_pos * self.pos_scale) * self.R_CART_POS
            + abs(cart_vel * self.vel_scale) * self.R_CART_VEL
            + abs(pole_vel * self.angvel_scale) * self.R_POLE_VEL
            + abs(action) * self.R_CONTROL
            + abs(action - self.last_action) * self.R_JERK
        )
        self.last_action = action
        reward = uprightness + self.R_SURVIVAL + penalty

        pole_deg = pole_angle * self.angle_scale
        urgency_penalty = -0.4 * pole_deg ** 2
        reward += urgency_penalty

        pole_vel_urgency = -0.15 * (pole_vel * self.angvel_scale) ** 2
        reward += pole_vel_urgency

        if (abs(pole_angle * self.angle_scale) < 0.15
                and abs(pole_vel * self.angvel_scale) < 1.0):
            reward += 0.5

        cart_pos_world = cart_pos * self.pos_scale
        pole_angle_world = pole_angle * self.angle_scale

        term_reason = "none"
        if abs(cart_pos_world) > self.rail_limit:
            term_reason = "rail"
        elif abs(pole_angle_world) > self.angle_limit:
            term_reason = "pole_angle"

        terminated = term_reason != "none"
        truncated = self.step_count >= self.max_episode_steps

        info = {"term_reason": term_reason}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        rail = p.getJointState(self.cart_body, self.joint_rail)
        hip = p.getJointState(self.cart_body, self.joint_hip)

        pole_angle = self._wrap_angle(hip[0])

        obs = np.array([
            np.clip(rail[0] / self.pos_scale, -1, 1),
            np.clip(rail[1] / self.vel_scale, -1, 1),
            np.clip(pole_angle / self.angle_scale, -1, 1),
            np.clip(hip[1] / self.angvel_scale, -1, 1),
        ], dtype=np.float32)
        return obs

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return ((theta + math.pi) % (2 * math.pi)) - math.pi

    def render(self):
        pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
