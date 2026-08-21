import ale_py
import gymnasium as gym
import numpy as np


OBSERVATION_SHAPE = (4, 84, 84)
FRAME_SKIP = 4
STACK_SIZE = 4
NOOP_MAX = 30
REPEAT_ACTION_PROBABILITY = 0.0
AGENT_ACTION_MEANINGS = ("NOOP", "RIGHT", "LEFT")
LIFE_LOSS_REWARD = -1.0


gym.register_envs(ale_py)


class NoopReset(gym.Wrapper):
    def __init__(self, env, noop_max=NOOP_MAX):
        super().__init__(env)
        if not isinstance(noop_max, int) or noop_max < 0:
            raise ValueError("noop_max must be a non-negative integer")

        action_meanings = self.unwrapped.get_action_meanings()
        if "NOOP" not in action_meanings:
            raise ValueError("environment does not provide a NOOP action")

        self.noop_action = action_meanings.index("NOOP")
        self.noop_max = noop_max

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        noop_count = int(
            self.unwrapped.np_random.integers(0, self.noop_max + 1)
        )

        for _ in range(noop_count):
            observation, _, terminated, truncated, step_info = self.env.step(
                self.noop_action
            )
            info = dict(info)
            info.update(step_info)
            if terminated or truncated:
                observation, info = self.env.reset(options=options)

        info = dict(info)
        info["noop_count"] = noop_count
        return observation, info


class EpisodicLifeFire(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        action_meanings = self.unwrapped.get_action_meanings()
        if "FIRE" not in action_meanings:
            raise ValueError("environment does not provide a FIRE action")

        self.fire_action = action_meanings.index("FIRE")
        self.life_terminated = False
        self.lives = 0

    def reset(self, *, seed=None, options=None):
        life_reset = self.life_terminated and self.lives > 0
        if life_reset:
            observation, _, terminated, truncated, info = self.env.step(
                self.fire_action
            )
        else:
            observation, info = self.env.reset(seed=seed, options=options)
            observation, _, terminated, truncated, fire_info = self.env.step(
                self.fire_action
            )
            info = dict(info)
            info.update(fire_info)

        if terminated or truncated:
            observation, info = self.env.reset(seed=seed, options=options)
            observation, _, terminated, truncated, fire_info = self.env.step(
                self.fire_action
            )
            info = dict(info)
            info.update(fire_info)
            life_reset = False
            if terminated or truncated:
                raise RuntimeError("Breakout terminated while serving the ball")

        self.life_terminated = False
        self.lives = self.unwrapped.ale.lives()
        info = dict(info)
        info.update({"auto_fire": True, "life_reset": life_reset})
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        lives = self.unwrapped.ale.lives()
        life_lost = lives < self.lives
        life_terminal = (
            life_lost and lives > 0 and not terminated and not truncated
        )

        if life_terminal:
            terminated = True
            self.life_terminated = True
        else:
            self.life_terminated = False

        applied_life_loss_reward = LIFE_LOSS_REWARD if life_lost else 0.0
        if life_lost:
            reward = LIFE_LOSS_REWARD

        self.lives = lives
        info = dict(info)
        info.update(
            {
                "life_lost": life_lost,
                "life_terminal": life_terminal,
                "life_loss_reward": applied_life_loss_reward,
                "auto_fire": False,
            }
        )
        return observation, reward, terminated, truncated, info


class RemoveFireAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        raw_action_meanings = self.unwrapped.get_action_meanings()
        try:
            self.action_map = tuple(
                raw_action_meanings.index(action_meaning)
                for action_meaning in AGENT_ACTION_MEANINGS
            )
        except ValueError as error:
            raise ValueError(
                "Breakout action space must provide NOOP, RIGHT and LEFT"
            ) from error

        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid agent action: {action}")
        return self.action_map[int(action)]

    def get_action_meanings(self):
        return list(AGENT_ACTION_MEANINGS)


class ClipReward(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.clip(reward, -1.0, 1.0))


def wrap_breakout_observation(env):
    env = NoopReset(env, noop_max=NOOP_MAX)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=0,
        frame_skip=FRAME_SKIP,
        screen_size=OBSERVATION_SHAPE[-1],
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = EpisodicLifeFire(env)
    env = RemoveFireAction(env)
    env = ClipReward(env)
    return gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)


def make_breakout_env(update, num_envs=4):
    if bool(update):
        return gym.make_vec(
            "ALE/Breakout-v5",
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[wrap_breakout_observation],
            frameskip=1,
            repeat_action_probability=REPEAT_ACTION_PROBABILITY,
        )

    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human",
        frameskip=1,
        repeat_action_probability=REPEAT_ACTION_PROBABILITY,
    )
    return wrap_breakout_observation(env)
