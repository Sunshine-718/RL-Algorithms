import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.optim import NAdam
from copy import deepcopy
from torch.distributions import Categorical
from common import ResidualBlock, ReplayBuffer, PPOAgentBase, NNBase

import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    capacity: int = 100000
    epoch: int = 10
    reward_scale: float = 1
    clip_coef: float = 0.25
    gaeLambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gp: float = 1


class DiscretePPO(NNBase):
    def __init__(self, lr, obs_dim, h_dim, action_dim, computes_grad=True, device='cpu'):
        super().__init__()
        self.policy = nn.Sequential(ResidualBlock(obs_dim, h_dim),
                                    ResidualBlock(h_dim, h_dim),
                                    ResidualBlock(h_dim, action_dim),
                                    nn.LogSoftmax(dim=-1))
        self.value = nn.Sequential(ResidualBlock(obs_dim, h_dim),
                                   ResidualBlock(h_dim, h_dim),
                                   ResidualBlock(h_dim, 1))
        self.opt = NAdam(self.parameters(), lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        torch.nn.init.constant_(self.policy[-2].linear.weight, 0)

        self.computes_grad(computes_grad)
        self.apply(self.init_weights)
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def actor(self, state):
        return self.policy(state)

    def critic(self, state):
        return self.value(state)

    def forward(self, state):
        return self.policy(state), self.value(state)

    def get_dist(self, state, action=None):
        log_probs = self.actor(state)
        if action is not None:
            return Categorical(log_probs.exp()), log_probs.gather(-1, action.long())
        return Categorical(log_probs.exp()), None


def get_actor_gradient(actor, state):
    state.requires_grad_(True)
    actor_output = actor(state)
    if isinstance(actor_output, tuple):
        grads = torch.autograd.grad(inputs=state,
                                    outputs=actor_output,
                                    grad_outputs=torch.ones_like(actor_output),
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
    def __init__(self, name, net, config):
        self.net: DiscretePPO = net
        self.target_net = deepcopy(net)
        self.target_net.computes_grad(False)
        self.buffer = ReplayBuffer(net.obs_dim, config.capacity, 1, net.device)

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
            log_probs = self.net.actor(state)
            action = torch.argmax(log_probs.view(-1)).item()
        else:
            action_dist, _ = self.net.get_dist(state)
            action = action_dist.sample()
            if action.size() == 1:
                action = action.item()
            else:
                action = action.squeeze().cpu().numpy()
        return action

    def step(self, batch_size=256):
        self.net.eval()
        states, actions, rewards, next_states, terminated, truncated = self.buffer.retrive_all()
        with torch.no_grad():
            values = self.net.critic(states).reshape(-1)
            next_values = self.net.critic(next_states).reshape(-1)
            advantages = self.GAE(self.discount, self.gaeLambda, rewards.reshape(-1),
                                  values, next_values, terminated).reshape(-1, 1)
            td_target = advantages + values.reshape(-1, 1)
            _, old_log_probs = self.net.get_dist(states, actions)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset = TensorDataset(states, actions, old_log_probs, advantages, td_target)
        loader = DataLoader(dataset, batch_size, True)
        self.net.train()
        for _ in range(self.epoch):
            for state, action, old_logp, adv, td_targ in loader:
                self.net.opt.zero_grad()
                dist, log_probs = self.net.get_dist(state, action)
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
    env = env = gym.make("CartPole-v1", render_mode='human' if not update else None).unwrapped
    # env = RescaleAction(env, -1, 1)
    ac = DiscretePPO(3e-4, env.observation_space.shape[0], 128, env.action_space.n, True, device)
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
    for i in iterator:
        state = env.reset()[0]
        episode_reward_sum = 0
        j = 0
        while True:
            j += 1
            action = agent.action(state, not update)
            next_state, reward, terminated, truncated, _ = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = 2 * r1 + r2
            if bool(update):
                agent.store(state, action, reward, next_state, terminated, truncated)
            episode_reward_sum += reward
            state = next_state
            if terminated or truncated or j > max_steps:
                break
        if bool(update) and len(agent.buffer) > 100:
            agent.step()
        reward_container.append(episode_reward_sum)
        avg[i % interval] = episode_reward_sum
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
            f'episode reward: {episode_reward_sum: .0f}, avg: {res: .0f}, best avg: {best_avg: .0f}, episode_length: {j}')
    env.close()
