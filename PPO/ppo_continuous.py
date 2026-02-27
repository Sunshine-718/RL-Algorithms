import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.optim import NAdam, AdamW
from copy import deepcopy
from torch.distributions import Beta

import gymnasium as gym
from gymnasium.wrappers import RescaleAction, NormalizeObservation, NormalizeReward, RecordEpisodeStatistics
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from common import ResidualBlock, ReplayBuffer, PPOAgentBase, NNBase, symlog, symexp
from dataclasses import dataclass, asdict


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    capacity: int = 100000
    epoch: int = 5
    reward_scale: float = 1
    clip_coef: float = 0.25
    gaeLambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gp: float = 0


class ContinuousPPO(NNBase):
    def __init__(self, lr, obs_dim, h_dim, action_dim, action_limit=1., computes_grad=True, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(ResidualBlock(obs_dim, h_dim),
                                    ResidualBlock(h_dim, h_dim))
        self.b_alpha = nn.Sequential(ResidualBlock(h_dim, h_dim),
                                     ResidualBlock(h_dim, action_dim))
        self.b_beta = deepcopy(self.b_alpha)
        self.value = nn.Sequential(ResidualBlock(obs_dim, h_dim),
                                   ResidualBlock(h_dim, h_dim),
                                   ResidualBlock(h_dim, h_dim),
                                   ResidualBlock(h_dim, 1))
        # self.value = nn.Linear(obs_dim, 1)
        self.opt = AdamW(self.parameters(), lr, weight_decay=0.0, eps=1e-5)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.device = device

        self.computes_grad(computes_grad)
        self.apply(self.init_weights)
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def actor_logit(self, state):
        hidden = self.hidden(state)
        return self.b_alpha(hidden), self.b_beta(hidden)

    def actor(self, state):
        alpha_logit, beta_logit = self.actor_logit(state)
        alpha_logit = torch.clamp(alpha_logit, -10, 5)
        beta_logit = torch.clamp(beta_logit, -10, 5)
        alpha = torch.exp(alpha_logit) + 1
        beta = torch.exp(beta_logit) + 1
        return alpha, beta

    def critic(self, state):
        return self.value(state)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    @torch.no_grad()
    def mean_action(self, state):
        alpha, beta = self.actor(state)
        action = (alpha / (alpha + beta)).cpu()
        if action.numel() == 1:
            return float(action)
        return action

    def get_dist_logp(self, state, action=None):
        alpha, beta = self.actor(state)
        dist = Beta(alpha, beta)
        if action is not None:
            return dist, dist.log_prob(action).sum(dim=-1, keepdim=True)
        return dist, None

    def transform(self, action):
        return (action - 0.5) * 2 * self.action_limit

    def inverse_transform(self, action):
        return action / (2 * self.action_limit) + 0.5


def get_actor_gradient(actor, state):
    state.requires_grad_(True)
    actor_output = actor(state)
    if isinstance(actor_output, tuple):
        alpha, beta = actor_output
        grads = torch.autograd.grad(inputs=state,
                                    outputs=[alpha, beta],
                                    grad_outputs=[torch.ones_like(alpha), torch.ones_like(beta)],
                                    create_graph=True,
                                    retain_graph=True)
    else:
        return None
    return grads[0]


def gradient_penalty(gradient, c_lambda=10):
    if gradient is None:
        return 0
    gradient = gradient.view(len(gradient), -1)
    grad_norm = gradient.norm(2, dim=1)
    return torch.mean(grad_norm - 0.1) ** 2 * c_lambda


class PPOAgent(PPOAgentBase):
    def __init__(self, name, net, config: Config):
        self.net = net
        self.buffer = ReplayBuffer(net.obs_dim, config.capacity, net.action_dim, net.device)

        self.device = self.net.device
        self.name = name
        self.action_dim = net.action_dim
        for key, value in asdict(config).items():
            setattr(self, key, value)

    @torch.no_grad()
    def action(self, state, deterministic=False):
        self.net.eval()
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        if deterministic:
            action = self.net.mean_action(state)
        else:
            action_dist, _ = self.net.get_dist_logp(state)
            action = action_dist.sample()
            if action.numel() == 1:
                action = action.item()
            else:
                action = action.squeeze().cpu().numpy()
        action = self.net.transform(action)
        return action

    def step(self, batch_size=64):
        self.net.eval()
        states, actions, rewards, next_states, terminated, truncated = self.buffer.retrive_all()
        actions = self.net.inverse_transform(actions)
        with torch.no_grad():
            values = symexp(self.net.critic(states).reshape(-1))
            next_values = symexp(self.net.critic(next_states).reshape(-1))
            advantages = self.GAE(self.discount, self.gaeLambda, rewards.reshape(-1), values,
                                  next_values, terminated, truncated).reshape(-1, 1)
            td_target = symlog(advantages + values.reshape(-1, 1))
            _, log_probs = self.net.get_dist_logp(states, actions)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = TensorDataset(states, actions, log_probs, advantages, td_target)
        loader = DataLoader(dataset, batch_size, True)
        self.net.train()
        for _ in range(self.epoch):
            for state, action, old_logp, adv, td_targ in loader:
                self.net.opt.zero_grad()
                dist, log_probs = self.net.get_dist_logp(state, action)
                ratio = torch.exp(log_probs - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * adv
                actor_loss = -torch.mean(torch.min(surr1, surr2))
                if self.gp > 0:
                    grads = get_actor_gradient(self.net.actor, state)
                    gp = gradient_penalty(grads, self.gp * 10)
                    actor_loss += gp

                if self.ent_coef != 0:
                    entropy = dist.entropy().sum(dim=-1).mean()
                    entropy_loss = -self.ent_coef * entropy
                else:
                    entropy_loss = 0

                new_value = self.net.critic(state)
                critic_loss = F.smooth_l1_loss(new_value, td_targ)

                loss = actor_loss + self.vf_coef * critic_loss + entropy_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.net.opt.step()
        self.buffer.reset()


if __name__ == "__main__":
    update = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make("InvertedPendulum-v5", render_mode='human' if not update else None)
    env = RescaleAction(env, -1, 1)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    ac = ContinuousPPO(1e-4, env.observation_space.shape[0], 128, env.action_space.shape[0], 1, True, device=device)
    config = Config()
    agent = PPOAgent('test', ac, config)
    # agent.load()
    reward_container = []
    Loss = []
    td_error = []
    max_steps = 1000
    interval = 10
    avg = np.zeros(interval)
    best_avg = -float('inf')
    res = 0
    iterator = tqdm(range(10000))
    plt.ion()

    # scaler = RewardScaler()
    for i in iterator:
        state = env.reset()[0]
        # scaler.reset()
        # state = ac.norm(state, bool(update))
        j = 0
        while True:
            j += 1
            action = agent.action(state, not update)
            next_state, reward, terminated, truncated, info = env.step(action)
            # next_state = ac.norm(next_state, bool(update))
            # reward = scaler(reward)
            if bool(update):
                agent.store(state, action, reward, next_state, terminated, truncated)
            state = next_state
            if terminated or truncated or j > max_steps:
                break
        if bool(update) and len(agent.buffer) > 2048:
            agent.step()
        reward_container.append(info['episode']['r'])
        avg[i % interval] = info['episode']['r']
        agent.save()
        if i % interval == 0 and i != 0:
            plt.clf()
            plt.plot(reward_container, label='Reward')
            plt.title(f'Reward: {reward_container[-1]}')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.pause(0.1)
            res = np.mean(avg)
            if res > best_avg:
                best_avg = res
        iterator.set_description(
            f'episode reward: {info['episode']['r']: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}')
    env.close()
