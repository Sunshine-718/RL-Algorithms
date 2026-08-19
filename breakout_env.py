import ale_py
import gymnasium as gym


OBSERVATION_SHAPE = (4, 84, 84)
FRAME_SKIP = 4
STACK_SIZE = 4
REPEAT_ACTION_PROBABILITY = 0.0


gym.register_envs(ale_py)


class FireReset(gym.Wrapper):
    def __init__(self, env, terminal_on_life_loss=False):
        super().__init__(env)
        action_meanings = self.unwrapped.get_action_meanings()
        if "FIRE" not in action_meanings:
            raise ValueError("environment does not provide a FIRE action")
        self.fire_action = action_meanings.index("FIRE")
        self.terminal_on_life_loss = bool(terminal_on_life_loss)
        self.life_terminated = False
        self.lives = 0

    def reset(self, **kwargs):
        if self.life_terminated and self.lives > 0:
            observation, _, terminated, truncated, info = self.env.step(
                self.fire_action
            )
            self.life_terminated = False
            if not terminated and not truncated:
                info = dict(info)
                info["life_reset"] = True
                self.lives = self.unwrapped.ale.lives()
                return observation, info

        observation, info = self.env.reset(**kwargs)
        observation, _, terminated, truncated, step_info = self.env.step(
            self.fire_action
        )
        if terminated or truncated:
            observation, info = self.env.reset(**kwargs)
        else:
            info = dict(info)
            info.update(step_info)
        info = dict(info)
        info["life_reset"] = False
        self.life_terminated = False
        self.lives = self.unwrapped.ale.lives()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        lives = self.unwrapped.ale.lives()
        life_lost = lives < self.lives
        life_terminal = (
            self.terminal_on_life_loss
            and life_lost
            and lives > 0
            and not terminated
        )
        auto_fire = (
            life_lost
            and not life_terminal
            and lives > 0
            and not terminated
            and not truncated
        )

        if auto_fire:
            (
                fire_observation,
                fire_reward,
                fire_terminated,
                fire_truncated,
                fire_info,
            ) = self.env.step(self.fire_action)
            observation = fire_observation
            reward = float(reward) + float(fire_reward)
            terminated = terminated or fire_terminated
            truncated = truncated or fire_truncated
            info = dict(info)
            info.update(fire_info)
            lives = self.unwrapped.ale.lives()

        if life_terminal:
            terminated = True
            self.life_terminated = not truncated
        else:
            self.life_terminated = False

        info = dict(info)
        info.update(
            {
                "life_lost": life_lost,
                "life_terminal": life_terminal,
                "auto_fire": auto_fire,
            }
        )
        self.lives = lives
        return observation, reward, terminated, truncated, info


def wrap_breakout_observation(env, terminal_on_life_loss=False):
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=FRAME_SKIP,
        screen_size=OBSERVATION_SHAPE[-1],
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FireReset(
        env, terminal_on_life_loss=terminal_on_life_loss
    )
    return gym.wrappers.FrameStackObservation(env, stack_size=STACK_SIZE)


def wrap_breakout_training_observation(env):
    return wrap_breakout_observation(env, terminal_on_life_loss=True)


def make_breakout_env(update, num_envs=4):
    if bool(update):
        return gym.make_vec(
            "ALE/Breakout-v5",
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[wrap_breakout_training_observation],
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
