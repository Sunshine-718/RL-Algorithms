import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam, SGD
import gymnasium as gym
import numpy as np
from pathlib import Path


LOG_ALPHA_MIN = -20.0
LOG_ALPHA_MAX = 20.0


def _unwrap_env(env):
    return env.unwrapped


def make_train_test_env(env_id, update, num_envs=16, unwrap=False,
                        rescale_action=False, **kwargs):
    if bool(update):
        env = gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode="async",
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.DISABLED,
            },
            wrappers=[_unwrap_env] if unwrap else [],
            **kwargs,
        )
        if rescale_action:
            env = gym.wrappers.vector.RescaleAction(env, -1, 1)
        return env

    env = gym.make(env_id, render_mode="human", **kwargs)
    if unwrap:
        env = env.unwrapped
    if rescale_action:
        env = gym.wrappers.RescaleAction(env, -1, 1)
    return env


def single_spaces(env, update):
    if bool(update):
        return env.single_observation_space, env.single_action_space
    return env.observation_space, env.action_space


def reset_env(env, update):
    state, _ = env.reset()
    if bool(update):
        return state
    return np.expand_dims(state, axis=0)


def step_env(env, actions, update):
    if bool(update):
        return env.step(actions)
    next_state, reward, terminated, truncated, info = env.step(actions[0])
    return (
        np.expand_dims(next_state, axis=0),
        np.asarray([reward], dtype=np.float32),
        np.asarray([terminated], dtype=bool),
        np.asarray([truncated], dtype=bool),
        info,
    )


def reset_done_envs(env, next_states, done, update):
    if not np.any(done):
        return next_states
    if bool(update):
        next_states, _ = env.reset(
            options={"reset_mask": done.astype(np.bool_)},
        )
        return next_states
    next_state, _ = env.reset()
    return np.expand_dims(next_state, axis=0)


def flush_episode(agent, transitions):
    for transition in transitions:
        agent.cache(*transition)
    agent.process()
    transitions.clear()


def store_n_step_transition(agent, transition_cache, force=False):
    if not transition_cache:
        return False
    if len(transition_cache) < agent.n_step and not force:
        return False

    horizon = min(agent.n_step, len(transition_cache))
    reward = sum(
        transition_cache[index][2] * agent.discount ** index
        for index in range(horizon)
    )
    state, action = transition_cache[0][:2]
    next_state, terminated, truncated = transition_cache[horizon - 1][3:]
    agent.buffer.store(
        state,
        action,
        reward,
        next_state,
        terminated,
        truncated,
        horizon,
    )
    transition_cache.pop(0)
    return True


def discrete_temperature_loss(
    log_alpha, log_probabilities, probabilities, target_entropy
):
    entropy_error = (log_probabilities + target_entropy).detach()
    return -(
        probabilities.detach() * log_alpha * entropy_error
    ).sum(dim=-1).mean()


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    """Pairwise quantile Huber loss for prediction and target distributions."""
    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    weight = torch.abs(
        tau.unsqueeze(-1) - (td_error.detach() < 0).float()
    )
    return (weight * huber).mean()


def weighted_quantile_huber_loss(pred, target, target_weight, tau, kappa=1.0):
    """Quantile Huber loss for a weighted empirical target distribution."""
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError("pred and target must have shape [batch, quantiles]")
    if target.shape != target_weight.shape:
        raise ValueError("target and target_weight must have the same shape")
    if pred.shape[0] != target.shape[0]:
        raise ValueError("pred and target batch sizes must match")
    if tau.shape[-1] != pred.shape[-1]:
        raise ValueError("tau must contain one value per predicted quantile")

    td_error = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_error = td_error.abs()
    huber = torch.where(
        abs_error <= kappa,
        0.5 * td_error.pow(2),
        kappa * (abs_error - 0.5 * kappa),
    )
    quantile_weight = torch.abs(
        tau.unsqueeze(-1) - (td_error.detach() < 0).float()
    )
    return (
        quantile_weight * huber * target_weight.unsqueeze(1)
    ).sum(dim=-1).mean()


class QuantileEmbedding(nn.Module):
    def __init__(self, quantile_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(quantile_dim, out_dim),
                                 nn.LayerNorm(out_dim),
                                 nn.SiLU(True),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LayerNorm(out_dim),
                                 nn.SiLU(True))
        self.quantile_dim = quantile_dim + 1

    def forward(self, tau):
        quantile_embedding = torch.arange(1, self.quantile_dim, device=tau.device) * \
            torch.pi * tau.unsqueeze(-1)  # [batch_size, num_quantiles, 64]
        quantile_embedding = torch.cos(quantile_embedding)
        return self.net(quantile_embedding)


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return torch.sigmoid(self.gate(x)) * self.proj(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.):
        super().__init__()
        self.glu = GLU(in_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim)
        self.norm = nn.RMSNorm(in_dim, 1e-5)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.residual = in_dim == out_dim

    def forward(self, x):
        residual = 0
        if self.residual:
            residual = x
        x = self.norm(x)
        x = self.glu(x)
        x = self.linear(x)
        return self.dropout(x) + residual


class NNBase(nn.Module):
    def configure_optimizer(self, weight_decay, learning_rate, betas=(0.9, 0.999)):
        return NAdam(self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay, decoupled_weight_decay=True)

    def computes_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def save(self, path=None, extra_state=None):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = self.state_dict()
            if extra_state is not None:
                checkpoint = {
                    'model_state_dict': checkpoint,
                    **extra_state,
                }
            torch.save(checkpoint, path)

    def load(self, path=None):
        try:
            if path is None:
                return False
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f'Checkpoint not found: {path}')
            return False
        finally:
            self.to(self.device)
        return checkpoint


class DQNAgentBase:
    @property
    def n_step(self):
        return self._n_step

    @n_step.setter
    def n_step(self, val):
        assert val >= 1 and isinstance(val, int)
        self._n_step = val
        self.buffer.n_step = val

    def cache(self, state, action, reward, next_state, terminated, truncated):
        reward = reward * getattr(self, 'reward_scale', 1.0)
        self.buffer.cache_transition(
            state, action, reward, next_state, terminated, truncated
        )

    def process(self):
        self.buffer.process()

    def save(self, model='last'):
        path = f'{self.name}_{model}.pt'
        if self.params is not None:
            path = f'{self.params}/{path}'
        extra_state = None
        if hasattr(self, '_alpha'):
            extra_state = {'log_alpha': self._alpha.detach().cpu()}
        self.net.save(path, extra_state)

    def load(self, model='last', required=False):
        path = f'{self.name}_{model}.pt'
        if self.params is not None:
            path = f'{self.params}/{path}'
        checkpoint = self.net.load(path)
        if checkpoint is False:
            if required:
                raise FileNotFoundError(path)
            return False
        if hasattr(self, '_alpha') and 'log_alpha' in checkpoint:
            with torch.no_grad():
                self._alpha.copy_(
                    checkpoint['log_alpha'].to(
                        device=self._alpha.device,
                        dtype=self._alpha.dtype,
                    )
                )
        self.soft_update(tau=1)
        return True

    def decay_noise(self, zero_noise=False):
        self.noise = max(self.min_noise, self.noise * self.decay) * (1 - bool(zero_noise))

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class SoftDQNAgentBase(DQNAgentBase):
    def configure_alpha(self, initial_alpha, *, lr=1e-2):
        initial_alpha = float(initial_alpha)
        if not np.isfinite(initial_alpha) or initial_alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        lr = float(lr)
        if not np.isfinite(lr) or lr <= 0.0:
            raise ValueError("alpha learning rate must be finite and positive")

        self._alpha = torch.tensor(
            [np.log(initial_alpha)], dtype=torch.float32, requires_grad=True
        )
        self._validate_alpha_()
        self.alpha_opt = SGD([self._alpha], lr=lr)
        return self

    @property
    def alpha(self):
        self._validate_alpha_()
        return float(self._alpha.detach().exp().item())

    @alpha.setter
    def alpha(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("alpha must be finite and positive")
        log_value = torch.tensor(
            [np.log(value)], dtype=self._alpha.dtype,
            device=self._alpha.device,
        )
        stored_value = log_value.exp()
        if (
            not torch.isfinite(log_value).all()
            or not torch.isfinite(stored_value).all()
            or not torch.all(stored_value > 0.0)
        ):
            raise ValueError("alpha must fit its parameter dtype")
        if not torch.all(
            (log_value >= LOG_ALPHA_MIN) & (log_value <= LOG_ALPHA_MAX)
        ):
            raise ValueError(
                "log_alpha must stay within the numerical safety range "
                f"[{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}]"
            )
        with torch.no_grad():
            self._alpha.copy_(log_value)

    @torch.no_grad()
    def _validate_alpha_(self):
        if not torch.isfinite(self._alpha).all():
            raise ValueError("log_alpha must be finite")
        alpha = self._alpha.exp()
        if not torch.isfinite(alpha).all() or not torch.all(alpha > 0.0):
            raise ValueError("alpha must be finite and positive")
        if not torch.all(
            (self._alpha >= LOG_ALPHA_MIN)
            & (self._alpha <= LOG_ALPHA_MAX)
        ):
            raise ValueError(
                "log_alpha must stay within the numerical safety range "
                f"[{LOG_ALPHA_MIN}, {LOG_ALPHA_MAX}]"
            )

    def load(self, model='last', required=False):
        previous_alpha = self._alpha.detach().clone()
        try:
            loaded = super().load(model, required)
            self._validate_alpha_()
        except Exception:
            with torch.no_grad():
                self._alpha.copy_(previous_alpha)
            raise
        return loaded

    def _update_alpha(self, q_values):
        if q_values.ndim == 3:
            q_values = q_values.mean(dim=-1)
        elif q_values.ndim != 2:
            raise ValueError(
                "q_values must have shape [batch, actions] or "
                "[batch, actions, quantiles]"
            )

        self._validate_alpha_()
        q_values = q_values.detach().cpu()
        log_probabilities = torch.log_softmax(
            q_values / self.alpha, dim=-1
        )
        probabilities = log_probabilities.exp()
        entropy = -(
            probabilities * log_probabilities
        ).sum(dim=-1).mean()

        self.alpha_opt.zero_grad()
        alpha_loss = discrete_temperature_loss(
            self._alpha,
            log_probabilities,
            probabilities,
            self.target_entropy,
        )
        if not torch.isfinite(alpha_loss):
            raise FloatingPointError("alpha loss must be finite")
        alpha_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_([self._alpha], 0.1)
        if not torch.isfinite(grad_norm):
            self.alpha_opt.zero_grad()
            raise FloatingPointError("alpha gradient must be finite")
        previous_alpha = self._alpha.detach().clone()
        self.alpha_opt.step()
        try:
            self._validate_alpha_()
        except ValueError:
            with torch.no_grad():
                self._alpha.copy_(previous_alpha)
            self.alpha_opt.zero_grad()
            raise
        return entropy

    def training_metrics(self, loss=None, entropy=None):
        if isinstance(loss, torch.Tensor):
            loss = float(loss.detach().item())
        elif loss is not None:
            loss = float(loss)
        if isinstance(entropy, torch.Tensor):
            entropy = float(entropy.detach().item())
        elif entropy is not None:
            entropy = float(entropy)
        metrics = {
            "updated": loss is not None,
            "loss": loss,
            "alpha": self.alpha,
            "entropy": entropy,
            "target_entropy": float(self.target_entropy),
            "buffer_size": len(self.buffer),
            "buffer_capacity": int(self.buffer.capacity),
        }
        self.last_training_metrics = metrics
        return metrics

    @staticmethod
    def format_training_metrics(metrics):
        loss = metrics["loss"]
        entropy = metrics["entropy"]
        loss_text = "n/a" if loss is None else f"{loss:.4f}"
        entropy_text = "n/a" if entropy is None else f"{entropy:.3f}"
        return (
            f"loss: {loss_text}, alpha: {metrics['alpha']:.4f}, "
            f"H: {entropy_text}/{metrics['target_entropy']:.3f}, "
            f"replay: {metrics['buffer_size']:,}/"
            f"{metrics['buffer_capacity']:,}"
        )
