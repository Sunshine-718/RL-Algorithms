from dataclasses import dataclass


@dataclass
class Config:
    discount: float = 0.99
    discount_lambda: float = 0.95
    capacity: int = 100_000
    batch_size: int = 16
    sequence_length: int = 32
    imagination_horizon: int = 15
    prefill: int = 1_000
    train_every: int = 5
    train_steps: int = 1
    total_steps: int = 1_000_000
    model_lr: float = 3e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    grad_clip: float = 100.0
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 1.0
    reconstruction_scale: float = 1.0
    reward_scale: float = 1.0
    continue_scale: float = 1.0
    discrete_entropy: float = 1e-3
    continuous_entropy: float = 1e-4
    slow_target_update: int = 100
    hidden_state_dim: int = 200
    stochastic_dim: int = 32
    stochastic_classes: int = 32
    rssm_mlp_dim: int = 200
    hidden_dim: int = 400
    cnn_depth: int = 48
    adam_eps: float = 1e-5
    weight_decay: float = 1e-6
    params: str = "./params"
    eval_episodes: int = 3
    seed: int = 0
