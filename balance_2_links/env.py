import math
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


URDF_PATH = os.path.join(os.path.dirname(__file__), "cartpole_double.urdf")


class DoublePendulumCartEnv(gym.Env):
    """Custom Gymnasium env: cart on a rail with an asymmetric 2-link pendulum.

    Action space: continuous force (N) applied to the cart along the rail (1D).
    Why continuous: it's a more natural fit for a physics sim and SB3 PPO handles
    it well; discrete left/right would force bang-bang control which is suboptimal
    for this asymmetric plant.

    Observation space (6 values, normalized ~[-1, 1]):
      [cart_pos, cart_vel, hip_angle, hip_angvel, elbow_angle, elbow_angvel]

    Reward: dense — uprightness of both links + survival bonus - movement/effort penalty.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 500):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # ------------------------------------------------------------------ #
        #  Action space: continuous 1D force on cart (positive = right)      #
        # ------------------------------------------------------------------ #
        self.max_force = 25.0
        self.action_space = spaces.Box(
            low=-self.max_force, high=self.max_force, shape=(1,), dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        #  Observation space: 6 values normalised to ~[-1, 1]                #
        # ------------------------------------------------------------------ #
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        #  REWARD WEIGHTS  ← tweak these to shape behaviour                 #
        # ------------------------------------------------------------------ #
        self.R_HIP = 1.0          # uprightness of link1 (cos term)
        self.R_ELBOW = 1.0        # uprightness of link2 (cos term)
        self.R_SURVIVAL = 0.1     # small bonus every timestep
        self.R_CART_POS = -0.5    # penalty for cart offset from center
        self.R_CART_VEL = -0.005  # nearly zero — fast bursts to catch falls are good
        self.R_HIP_VEL = -0.05    # penalty for hip angular velocity
        self.R_ELBOW_VEL = -0.05  # penalty for elbow angular velocity
        self.R_CONTROL = -0.03    # penalty for |action| (discourage large forces)
        self.R_JERK = -0.005      # light — prevents constant twitch without penalizing sharp saves

        # ------------------------------------------------------------------ #
        #  Physics & termination limits                                      #
        # ------------------------------------------------------------------ #
        self.rail_limit = 2.4    # cart derails beyond this (URDF limit is 2.5)
        self.angle_limit = 1.5   # ~86° from upright — any link past this = done

        # Observation normalisation scales
        self.pos_scale = self.rail_limit
        self.vel_scale = 10.0
        self.angle_scale = math.pi
        self.angvel_scale = 10.0

        # PyBullet internals (populated by _setup_sim)
        self.physics_client: int | None = None
        self.cart_body: int | None = None

        # Joint indices — order matches URDF joint declarations
        self.joint_rail = 0
        self.joint_hip = 1
        self.joint_elbow = 2

        self.last_action = 0.0

        self._setup_sim()

    # ------------------------------------------------------------------ #
    #  PyBullet initialisation                                           #
    # ------------------------------------------------------------------ #
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

        # Disable default motors on the passive joints so they swing freely
        # (setting force=0 on VELOCITY_CONTROL effectively disables the motor)
        for jid in (self.joint_hip, self.joint_elbow):
            p.setJointMotorControl2(self.cart_body, jid, p.VELOCITY_CONTROL, force=0)

        # Keep the cart joint motor disabled too — we'll send raw torque in step()
        p.setJointMotorControl2(self.cart_body, self.joint_rail, p.VELOCITY_CONTROL, force=0)

    # ------------------------------------------------------------------ #
    #  Gymnasium interface                                               #
    # ------------------------------------------------------------------ #
    CURRICULUM_FALLING_START_PROB = 0.3

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_action = 0.0

        # Small random perturbation around the upright equilibrium
        # (not a hanging-down start — so the agent can't exploit a single lucky trajectory)
        rail_pos = self.np_random.uniform(-0.1, 0.1)
        rail_vel = self.np_random.uniform(-0.5, 0.5)
        hip_angle_init = self.np_random.uniform(-0.2, 0.2)
        hip_vel_init = self.np_random.uniform(-0.3, 0.3)
        elbow_angle_init = self.np_random.uniform(-0.2, 0.2)
        elbow_vel_init = self.np_random.uniform(-0.3, 0.3)

        # Curriculum: 30% of episodes start with real angular velocity already
        # present, matching the regime where recovery currently fails (hip_vel ~2.5-4.0).
        if self.np_random.random() < self.CURRICULUM_FALLING_START_PROB:
            hip_vel_init = self.np_random.uniform(-4.0, 4.0)
            elbow_vel_init = self.np_random.uniform(-2.0, 2.0)

        p.resetJointState(self.cart_body, self.joint_rail, targetValue=rail_pos, targetVelocity=rail_vel)
        p.resetJointState(self.cart_body, self.joint_hip, targetValue=hip_angle_init, targetVelocity=hip_vel_init)
        p.resetJointState(self.cart_body, self.joint_elbow, targetValue=elbow_angle_init, targetVelocity=elbow_vel_init)

        p.stepSimulation()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_force, self.max_force)[0]

        # Apply continuous force to the cart's prismatic joint
        p.setJointMotorControl2(
            self.cart_body, self.joint_rail, p.TORQUE_CONTROL,
            force=float(action),
        )
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(1.0 / 240.0)
        self.step_count += 1

        obs = self._get_obs()
        cart_pos, cart_vel, hip_angle, hip_vel, elbow_angle, elbow_vel = obs

        # ---- reward ----
        # Weighted-sum uprightness: hip gets more weight since it's the dominant
        # failure mode (90% of terminations). Previously equal multiplicative.
        # u = cos(hip) * cos(elbow)  # old — equal multiplicative
        uprightness = (
            0.65 * math.cos(hip_angle * self.angle_scale)
            + 0.35 * math.cos(elbow_angle * self.angle_scale)
        )

        penalty = (
            abs(cart_pos * self.pos_scale) * self.R_CART_POS
            + abs(cart_vel * self.vel_scale) * self.R_CART_VEL
            + abs(hip_vel * self.angvel_scale) * self.R_HIP_VEL
            + abs(elbow_vel * self.angvel_scale) * self.R_ELBOW_VEL
            + abs(action) * self.R_CONTROL
            + abs(action - self.last_action) * self.R_JERK
        )
        self.last_action = action
        reward = uprightness + self.R_SURVIVAL + penalty

        # Quadratic urgency penalty: hip weighted more heavily.
        # Previously equal: -0.3 * hip_deg**2 - 0.3 * elbow_deg**2
        hip_deg = hip_angle * self.angle_scale
        elbow_deg = elbow_angle * self.angle_scale
        urgency_penalty = -0.4 * hip_deg**2 - 0.2 * elbow_deg**2
        reward += urgency_penalty

        # Velocity urgency: quadratic hip_vel penalty sharpens recovery gradient
        # at the moderate velocities where the policy currently freezes.
        hip_vel_urgency = -0.15 * (hip_vel * self.angvel_scale) ** 2
        reward += hip_vel_urgency

        # "Settled" bonus: only when both links are near-vertical AND slow.
        # The dense cos reward is happy with near-vertical oscillation; this
        # threshold bonus pushes the agent to actually converge at the top.
        if (abs(hip_angle * self.angle_scale) < 0.15
                and abs(elbow_angle * self.angle_scale) < 0.15
                and abs(hip_vel * self.angvel_scale) < 1.0
                and abs(elbow_vel * self.angvel_scale) < 1.0):
            reward += 0.5

        # ---- termination ----
        cart_pos_world = cart_pos * self.pos_scale
        hip_angle_world = hip_angle * self.angle_scale
        elbow_angle_world = elbow_angle * self.angle_scale

        term_reason = "none"
        if abs(cart_pos_world) > self.rail_limit:
            term_reason = "rail"
        elif abs(hip_angle_world) > self.angle_limit:
            term_reason = "hip_angle"
        elif abs(elbow_angle_world) > self.angle_limit:
            term_reason = "elbow_angle"

        terminated = term_reason != "none"
        truncated = self.step_count >= self.max_episode_steps

        info = {"term_reason": term_reason}
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        rail = p.getJointState(self.cart_body, self.joint_rail)
        hip = p.getJointState(self.cart_body, self.joint_hip)
        elbow = p.getJointState(self.cart_body, self.joint_elbow)

        hip_angle = self._wrap_angle(hip[0])
        elbow_angle = self._wrap_angle(elbow[0])

        obs = np.array([
            np.clip(rail[0] / self.pos_scale, -1, 1),
            np.clip(rail[1] / self.vel_scale, -1, 1),
            np.clip(hip_angle / self.angle_scale, -1, 1),
            np.clip(hip[1] / self.angvel_scale, -1, 1),
            np.clip(elbow_angle / self.angle_scale, -1, 1),
            np.clip(elbow[1] / self.angvel_scale, -1, 1),
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
