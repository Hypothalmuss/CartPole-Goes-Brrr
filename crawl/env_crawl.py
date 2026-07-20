import math
import os
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


URDF_PATH = os.path.join(os.path.dirname(__file__), "cartpole_crawl.urdf")


class CrawlPoleEnv(gym.Env):
    """Custom Gymnasium env: the same 2-link asymmetric pendulum as the balance
    task, but the rail is gone. The cart is a free body on the ground plane and
    the pole is the active agent -- it has to use hip/elbow torque to push and
    drag the cart forward, inchworm-style.

    Action space: continuous torque (Nm) on the hip and elbow joints (2D).
    Why continuous + torque control (not velocity): locomotion gaits emerge
    from force interactions with the ground, so the policy needs direct control
    over how hard the links push against the floor, not just their target speed.

    Observation space (8 values, normalized ~[-1, 1]):
      [cart_x, cart_vx, cart_z, hip_angle, hip_angvel, elbow_angle, elbow_angvel,
       tip_contact]

    Reward: dense -- forward progress on the cart is the dominant term, with a
    survival bonus, an energy penalty, a height penalty (don't fly), and a small
    bonus for planting link2's tip on the ground (encourages inchworm contact).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_episode_steps: int = 1000):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # ------------------------------------------------------------------ #
        #  Action space: continuous 2D torque [hip, elbow]                   #
        # ------------------------------------------------------------------ #
        self.max_torque_hip = 20.0
        self.max_torque_elbow = 15.0
        self.action_space = spaces.Box(
            low=np.array([-self.max_torque_hip, -self.max_torque_elbow], dtype=np.float32),
            high=np.array([self.max_torque_hip, self.max_torque_elbow], dtype=np.float32),
            dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        #  Observation space: 8 values normalised to ~[-1, 1]                #
        # ------------------------------------------------------------------ #
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32,
        )

        # ------------------------------------------------------------------ #
        #  REWARD WEIGHTS  <- tweak these to shape behaviour                 #
        # ------------------------------------------------------------------ #
        self.R_PROGRESS = 3.0       # forward cart velocity (dominant term)
        self.R_SURVIVAL = 0.0       # removed: was worth more than crawling itself
        self.R_ENERGY = -0.003      # penalty on |hip_torque| + |elbow_torque|
        self.R_HEIGHT = -10.0       # heavy penalty on cart_z -- must stay grounded
        self.R_TIP_CONTACT = 0.3    # bonus for tip-on-ground *while moving forward*
        self.R_GROUND_REACH = 0.5   # reward for getting link tips close to ground
        self.tip_contact_progress_threshold = 0.005  # m/s -- must be advancing to earn tip bonus

        # ------------------------------------------------------------------ #
        #  Physics & termination limits                                      #
        # ------------------------------------------------------------------ #
        self.max_cart_height = 0.15  # cart flying above this ends the episode immediately
        self.height_baseline = 0.05  # penalize ANY lift above this (barely off ground)

        # Observation normalisation scales
        self.pos_scale = 5.0
        self.vel_scale = 5.0
        self.height_scale = 0.5
        self.angle_scale = math.pi
        self.angvel_scale = 10.0

        # PyBullet internals (populated by _setup_sim)
        self.physics_client: int | None = None
        self.cart_body: int | None = None
        self.plane_id: int | None = None

        # Joint indices -- order matches URDF joint declarations. Note there is
        # no rail joint here: the cart is the free base link (index -1), so hip
        # is joint 0 and elbow is joint 1 (vs. 1 and 2 in the balance task).
        self.joint_hip = 0
        self.joint_elbow = 1
        self.link2_index = self.joint_elbow  # child link index == joint index

        self.prev_cart_x = 0.0
        self.dt = 1.0 / 240.0  # pybullet default sim timestep

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
        p.setTimeStep(self.dt)

        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0)

        # useFixedBase=False: the base link ("cart") is a free 6-DOF body, not
        # welded to the world. It stays on the ground via gravity + contact,
        # same as any other object you'd drop into the scene.
        self.cart_body = p.loadURDF(URDF_PATH, [0, 0, 0.06], useFixedBase=False)
        p.changeDynamics(self.cart_body, -1, lateralFriction=1.0)

        # Disable default motors on hip/elbow so we can drive them with raw
        # torque commands in step() (same trick as the rail joint in env.py).
        for jid in (self.joint_hip, self.joint_elbow):
            p.setJointMotorControl2(self.cart_body, jid, p.VELOCITY_CONTROL, force=0)

    # ------------------------------------------------------------------ #
    #  Gymnasium interface                                               #
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_count = 0

        # Start lying flat on the ground: cart at a random X near the origin,
        # hip/elbow near 0 (which, thanks to the URDF's rotated hip origin,
        # means "link stretched out horizontally" rather than "upright").
        cart_x_init = self.np_random.uniform(-0.3, 0.3)
        hip_angle_init = self.np_random.uniform(-0.05, 0.05)
        elbow_angle_init = self.np_random.uniform(-0.05, 0.05)

        p.resetBasePositionAndOrientation(
            self.cart_body, [cart_x_init, 0, 0.02], [0, 0, 0, 1],
        )
        p.resetBaseVelocity(self.cart_body, [0, 0, 0], [0, 0, 0])
        p.resetJointState(self.cart_body, self.joint_hip, targetValue=hip_angle_init, targetVelocity=0.0)
        p.resetJointState(self.cart_body, self.joint_elbow, targetValue=elbow_angle_init, targetVelocity=0.0)

        # Let the assembly settle onto the ground under gravity before the
        # episode starts counting, with motors free (force=0, set in _setup_sim).
        for _ in range(20):
            p.stepSimulation()

        cart_pos, _ = p.getBasePositionAndOrientation(self.cart_body)
        self.prev_cart_x = cart_pos[0]

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(
            action, self.action_space.low, self.action_space.high,
        )
        hip_torque, elbow_torque = float(action[0]), float(action[1])

        p.setJointMotorControl2(
            self.cart_body, self.joint_hip, p.TORQUE_CONTROL, force=hip_torque,
        )
        p.setJointMotorControl2(
            self.cart_body, self.joint_elbow, p.TORQUE_CONTROL, force=elbow_torque,
        )
        p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(self.dt)
        self.step_count += 1

        cart_pos, _ = p.getBasePositionAndOrientation(self.cart_body)
        cart_x, _, cart_z = cart_pos

        # ---- reward ----
        # Forward-progress is an actual velocity estimate (delta-x / dt), not a
        # raw position delta, so it stays on a sane scale regardless of dt.
        progress = (cart_x - self.prev_cart_x) / self.dt
        self.prev_cart_x = cart_x

        # Only count progress when grounded -- jumping forward shouldn't earn reward
        grounded = cart_z < self.height_baseline + 0.05
        effective_progress = progress if grounded else 0.0

        energy = abs(hip_torque) + abs(elbow_torque)
        height_over_baseline = max(0.0, cart_z - self.height_baseline)
        tip_contact = len(p.getContactPoints(
            bodyA=self.cart_body, bodyB=self.plane_id, linkIndexA=self.link2_index,
        )) > 0

        # Ground-proximity reward: encourage links to reach toward the ground
        # so they can generate contact forces for crawling.
        hip_link_state = p.getLinkState(self.cart_body, self.joint_hip, computeForwardKinematics=True)
        elbow_link_state = p.getLinkState(self.cart_body, self.joint_elbow, computeForwardKinematics=True)
        hip_tip_z = hip_link_state[0][2]  # world z of link1 COM
        elbow_tip_z = elbow_link_state[0][2]  # world z of link2 COM
        ground_reach = max(0.0, 0.15 - hip_tip_z) + max(0.0, 0.1 - elbow_tip_z)

        reward = (
            self.R_PROGRESS * effective_progress
            + self.R_ENERGY * energy
            + self.R_HEIGHT * height_over_baseline
            + (self.R_TIP_CONTACT if (tip_contact and effective_progress > self.tip_contact_progress_threshold) else 0.0)
            + self.R_GROUND_REACH * ground_reach
        )

        # ---- termination ----
        # No "fallen" state here -- every angle is valid, the pole is supposed
        # to be down on the ground. The only failure mode is flying off it.
        term_reason = "none"
        if cart_z > self.max_cart_height:
            term_reason = "flew"

        terminated = term_reason != "none"
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_obs(cart_pos=cart_pos, tip_contact=tip_contact)
        info = {"term_reason": term_reason, "cart_x": cart_x}
        return obs, reward, terminated, truncated, info

    def _get_obs(self, cart_pos=None, tip_contact: bool | None = None) -> np.ndarray:
        if cart_pos is None:
            cart_pos, _ = p.getBasePositionAndOrientation(self.cart_body)
        cart_x, _, cart_z = cart_pos
        cart_vel, _ = p.getBaseVelocity(self.cart_body)
        cart_vx = cart_vel[0]

        hip = p.getJointState(self.cart_body, self.joint_hip)
        elbow = p.getJointState(self.cart_body, self.joint_elbow)
        hip_angle = self._wrap_angle(hip[0])
        elbow_angle = self._wrap_angle(elbow[0])

        if tip_contact is None:
            tip_contact = len(p.getContactPoints(
                bodyA=self.cart_body, bodyB=self.plane_id, linkIndexA=self.link2_index,
            )) > 0

        obs = np.array([
            np.clip(cart_x / self.pos_scale, -1, 1),
            np.clip(cart_vx / self.vel_scale, -1, 1),
            np.clip(cart_z / self.height_scale, -1, 1),
            np.clip(hip_angle / self.angle_scale, -1, 1),
            np.clip(hip[1] / self.angvel_scale, -1, 1),
            np.clip(elbow_angle / self.angle_scale, -1, 1),
            np.clip(elbow[1] / self.angvel_scale, -1, 1),
            1.0 if tip_contact else 0.0,
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
