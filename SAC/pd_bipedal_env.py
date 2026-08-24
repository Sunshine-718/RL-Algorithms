import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker


PD_ENV_ID = 'BipedalWalkerPD-v0'
JOINT_LOWER_LIMITS = np.array([-0.8, -1.6, -0.8, -1.6], dtype=np.float32)
JOINT_UPPER_LIMITS = np.array([1.1, -0.1, 1.1, -0.1], dtype=np.float32)
DEFAULT_KP = np.array([2., 2., 2., 2.], dtype=np.float32)
DEFAULT_KD = np.array([0.15, 0.1, 0.15, 0.1], dtype=np.float32)


class PD:
    def __init__(self, kp, kd, lim1=-1., lim2=1.):
        self.kp = np.asarray(kp, dtype=np.float32)
        self.kd = np.asarray(kd, dtype=np.float32)
        self.lim1 = lim1
        self.lim2 = lim2

    @staticmethod
    def clamp(num, lim1, lim2):
        return np.clip(num, min(lim1, lim2), max(lim1, lim2))

    def update(self, target, var, velocity):
        target = np.asarray(target, dtype=np.float32)
        var = np.asarray(var, dtype=np.float32)
        velocity = np.asarray(velocity, dtype=np.float32)
        error = target - var
        # 计算P
        p = error * self.kp
        # 计算D，目标角每步都会变化，所以直接使用关节角速度
        d = -velocity * self.kd
        result = self.clamp(p + d, self.lim1, self.lim2)
        return result.astype(np.float32)


def normalized_action_to_target(action):
    # 把策略动作从[-1, 1]映射到关节限位
    action = np.clip(np.asarray(action, dtype=np.float32), -1., 1.)
    center = (JOINT_LOWER_LIMITS + JOINT_UPPER_LIMITS) / 2
    half_range = (JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS) / 2
    return center + action * half_range


class PDBipedalWalker(BipedalWalker):
    def __init__(self, render_mode=None, hardcore=False,
                 kp=DEFAULT_KP, kd=DEFAULT_KD):
        super().__init__(render_mode=render_mode, hardcore=hardcore)
        self.pd = PD(kp, kd)
        self.resetting = False

    def reset(self, *, seed=None, options=None):
        # 原环境reset会调用一次step(0)，初始化时不经过PD
        self.resetting = True
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.resetting = False

    def step(self, action):
        if self.resetting:
            return BipedalWalker.step(self, action)

        target = normalized_action_to_target(action)
        joint_angles = np.array([i.angle for i in self.joints], dtype=np.float32)
        joint_velocities = np.array([i.speed for i in self.joints], dtype=np.float32)
        motor_actions = self.pd.update(target, joint_angles, joint_velocities)
        observation, reward, terminated, truncated, info = super().step(motor_actions)
        info['pd_target_angles'] = target
        info['pd_joint_angles'] = joint_angles
        info['pd_motor_actions'] = motor_actions
        info['pd_mean_absolute_error'] = np.abs(target - joint_angles).mean()
        info['pd_saturation_rate'] = (np.abs(motor_actions) >= 1 - 1e-6).mean()
        return observation, reward, terminated, truncated, info


if PD_ENV_ID not in gym.registry:
    gym.register(
        id=PD_ENV_ID,
        entry_point=PDBipedalWalker,
        max_episode_steps=1600,
        reward_threshold=300,
    )


def make_pd_bipedal_env(render_mode=None, hardcore=False,
                         kp=DEFAULT_KP, kd=DEFAULT_KD,
                         max_episode_steps=1600):
    return gym.make(
        PD_ENV_ID, render_mode=render_mode, hardcore=hardcore,
        kp=kp, kd=kd, max_episode_steps=max_episode_steps,
    )
