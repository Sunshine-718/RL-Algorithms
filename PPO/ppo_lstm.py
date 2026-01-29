import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import NAdam
from dataclasses import dataclass
from torch.distributions import Categorical
import gymnasium as gym
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


@dataclass
class Config:
    discount: float = 0.99
    params: str = './params'
    capacity: int = 2048        # PPO通常收集固定长度轨迹，比如2048
    epoch: int = 10
    reward_scale: float = 1
    clip_coef: float = 0.2
    gaeLambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5  # 增加梯度裁剪阈值配置

    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    seq_len: int = 8            # BPTT 长度


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.Tanh())

    def forward(self, x):
        return self.net(x)


class RecurrentPPO(nn.Module):
    def __init__(self, lr, obs_dim, h_dim, action_dim, lstm_hidden_dim, lstm_layers, computes_grad=True, device='cpu'):
        super().__init__()
        self.embedding = nn.Sequential(Block(obs_dim, h_dim),
                                       Block(h_dim, h_dim))

        self.lstm = nn.LSTM(h_dim, lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        self.actor_head = nn.Sequential(Block(lstm_hidden_dim, h_dim),
                                        nn.Linear(h_dim, action_dim))
        self.critic_head = nn.Sequential(Block(lstm_hidden_dim, h_dim),
                                         nn.Linear(h_dim, 1))

        self.opt = NAdam(self.parameters(), lr, eps=1e-5)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.device = device

        self.computes_grad(computes_grad)
        self.apply(self.init_weights)
        self.to(device)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)

    def computes_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad_(requires_grad)

    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        try:
            if path is not None:
                self.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as _:
            print('Failed to load parameters.')
        finally:
            self.to(self.device)

    def get_feature_and_memory(self, state, lstm_state, terminated=None):
        """
        核心修复：如果提供了 terminated (在训练时)，则手动循环处理序列以正确 Mask 掉 hidden state。
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [Batch, 1, Dim]

        features = self.embedding(state)  # [Batch, Seq, Dim]

        # Inference 模式或无 done 信号，直接调用 LSTM 加速
        if terminated is None:
            self.lstm.flatten_parameters()
            output, (hn, cn) = self.lstm(features, lstm_state)
            return output, (hn, cn)

        # Training 模式：必须逐步处理以重置 Hidden State
        output = []
        h, c = lstm_state
        seq_len = features.shape[1]

        for t in range(seq_len):
            # 这里的 terminated 形状应该是 [Batch, Seq]
            # mask: 如果当前步是 done，则下一步的 hidden 应该重置为 0
            # 注意：传入的 terminated[t] 表示 t 时刻是否结束，影响的是 t+1 时刻的计算
            # 但 PyTorch LSTM 是 Current Input + Prev Hidden -> Current Output
            # 所以我们要 Mask 的是传入本步的 hidden (即上一步的输出)

            mask = 1.0 - terminated[:, t].view(1, -1, 1)  # [1, Batch, 1]
            h = h * mask
            c = c * mask

            input_t = features[:, t].unsqueeze(1)  # [Batch, 1, Dim]
            out_t, (h, c) = self.lstm(input_t, (h, c))
            output.append(out_t)

        output = torch.cat(output, dim=1)  # [Batch, Seq, Hidden]
        return output, (h, c)

    def get_dist_logp(self, state, lstm_state, action=None, terminated=None):
        output, _ = self.get_feature_and_memory(state, lstm_state, terminated)
        logits = self.actor_head(output)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if action is not None:
            if action.dim() == 3 and action.shape[-1] == 1:
                action = action.squeeze(-1)
            return dist, dist.log_prob(action)
        return dist, None

    def get_value(self, state, lstm_state, terminated=None):
        """单独计算 Value，用于 GAE"""
        output, _ = self.get_feature_and_memory(state, lstm_state, terminated)
        return self.critic_head(output)

    def action(self, state, lstm_state, deterministic=False):
        """Inference 专用接口"""
        state_t = torch.from_numpy(state).float().to(self.device).reshape(1, 1, -1)
        dist, _ = self.get_dist_logp(state_t, lstm_state)  # Inference 不传 terminated

        _, next_lstm_state = self.get_feature_and_memory(state_t, lstm_state)

        if deterministic:
            action = dist.probs.argmax(dim=-1).item()
        else:
            action = dist.sample().item()
        return action, next_lstm_state


class RecurrentReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, lstm_hidden_dim, num_layers, device='cpu'):
        self.state = torch.empty((capacity, state_dim), dtype=torch.float32, device=device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.terminated = torch.empty((capacity, 1), dtype=torch.float32, device=device)

        # 存储的是每一步开始时的 Hidden State (Pre-update)
        self.hidden_h = torch.empty((capacity, num_layers, lstm_hidden_dim), dtype=torch.float32, device=device)
        self.hidden_c = torch.empty_like(self.hidden_h)

        self.counter = 0
        self.device = device
        self.capacity = capacity
        self.lstm_hidden_dim = lstm_hidden_dim

    def __len__(self):
        return self.counter

    def reset(self):
        self.counter = 0
        return self

    @torch.no_grad()
    def store(self, state, action, reward, terminated, lstm_state):
        if self.counter >= self.capacity:
            return

        idx = self.counter
        self.counter += 1

        self.state[idx] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.action[idx] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.reward[idx] = float(reward)
        self.terminated[idx] = float(terminated)

        # 存储 hidden state [Layers, Batch=1, Dim] -> squeeze(1) -> [Layers, Dim]
        self.hidden_h[idx] = lstm_state[0].squeeze(1).detach()
        self.hidden_c[idx] = lstm_state[1].squeeze(1).detach()

    def make_batch_iterator(self, batch_size, seq_len):
        """
        创建一个生成器，遍历所有有效数据 (On-Policy)
        """
        # 可用的最大索引 (减去 seq_len 确保不会越界)
        max_idx = self.counter - seq_len
        if max_idx <= 0:
            return

        # 生成所有可能的起始点索引并打乱
        indices = np.arange(max_idx)
        np.random.shuffle(indices)

        for start_pos in range(0, len(indices), batch_size):
            batch_indices = indices[start_pos: start_pos + batch_size]

            states, actions, rewards, terminateds = [], [], [], []
            hidden_hs, hidden_cs = [], []

            for idx in batch_indices:
                sl = slice(idx, idx + seq_len)
                states.append(self.state[sl])
                actions.append(self.action[sl])
                # 注意：terminated 也要切片，用于 Masking
                terminateds.append(self.terminated[sl])

                # 只需要序列开始时的 Hidden State
                hidden_hs.append(self.hidden_h[idx])
                hidden_cs.append(self.hidden_c[idx])

            # 堆叠数据
            # Hidden: [Batch, Layers, Dim] -> [Layers, Batch, Dim] (LSTM 格式)
            b_h0 = torch.stack(hidden_hs).transpose(0, 1)
            b_c0 = torch.stack(hidden_cs).transpose(0, 1)

            yield (
                torch.stack(states),        # [Batch, Seq, Obs]
                torch.stack(actions),       # [Batch, Seq, Act]
                torch.stack(terminateds),   # [Batch, Seq, 1]
                (b_h0, b_c0),               # Initial Hidden for sequence
                batch_indices               # 返回索引以便外部获取对应的 GAE/Returns
            )

    def retrive_all(self):
        length = self.counter
        return (self.state[:length], self.action[:length], self.reward[:length],
                self.terminated[:length], (self.hidden_h[:length], self.hidden_c[:length]))


class PPOAgentLSTM:
    def __init__(self, name, net, config: Config):
        self.net = net
        self.buffer = RecurrentReplayBuffer(net.obs_dim, config.capacity, 1,
                                            config.lstm_hidden_dim, config.lstm_layers, net.device)
        self.device = self.net.device
        self.name = name
        self.config = config
        self.reset_lstm_state()

    def reset_lstm_state(self):
        h = torch.zeros(self.config.lstm_layers, 1, self.config.lstm_hidden_dim, device=self.device)
        c = torch.zeros(self.config.lstm_layers, 1, self.config.lstm_hidden_dim, device=self.device)
        self.current_lstm_state = (h, c)

    def store(self, state, action, reward, terminated, truncated):
        # 存储的是执行 Action 之前的 State 和 Hidden State
        self.buffer.store(state, action, reward, terminated, self.current_lstm_state)

        # Store 之后如果 Done，重置 Agent 的 LSTM 状态供下一次 Step 使用
        if terminated or truncated:
            self.reset_lstm_state()

    @torch.no_grad()
    def action(self, state, deterministic=False):
        self.net.eval()
        action, next_lstm_state = self.net.action(state, self.current_lstm_state, deterministic)
        self.current_lstm_state = next_lstm_state
        return action

    def step(self, batch_size=32):
        self.net.eval()

        # 1. 准备数据
        states, actions, rewards, terminateds, (hs, cs) = self.buffer.retrive_all()

        # hs: [Total, Layers, Dim] -> [Layers, Total, Dim]
        b_h_all = hs.transpose(0, 1)
        b_c_all = cs.transpose(0, 1)

        with torch.no_grad():
            # [Total, 1, Obs]
            states_in = states.unsqueeze(1)

            # --- [Fix 1] ---
            # 计算 Values: 结果 squeeze 掉 seq_len 维度 -> [2048, 1] -> [2048]
            values = self.net.get_value(states_in, (b_h_all, b_c_all), terminated=None).squeeze(-1)

            # 计算 Old Log Probs
            dist, _ = self.net.get_dist_logp(states_in, (b_h_all, b_c_all), action=None, terminated=None)

            # --- [Fix 2] ---
            # 不要 squeeze action，保持 [2048, 1] 以匹配 dist 的 [2048, 1]
            # 结果 [2048, 1] -> squeeze(-1) -> [2048]
            old_log_probs = dist.log_prob(actions).squeeze(-1).detach()

        # 3. 计算 GAE
        # 确保 rewards 和 values 都是 [Total, 1] 或者都是 [Total]
        # 这里统一用 [Total, 1] 进行计算，方便处理
        rewards = rewards.view(-1, 1)
        terminateds = terminateds.view(-1, 1)
        values = values.view(-1, 1)  # [2048, 1]

        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - terminateds[t]
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - terminateds[t]
                nextvalues = values[t+1]

            delta = rewards[t] + self.config.discount * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + self.config.discount * self.config.gaeLambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values

        # 归一化 Advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 展平为 [Total] 以便存入列表
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)
        old_log_probs = old_log_probs.view(-1)  # 确保是 [Total]

        # 4. 训练循环 (On-Policy)
        self.net.train()
        seq_len = self.config.seq_len

        for _ in range(self.config.epoch):
            data_loader = self.buffer.make_batch_iterator(batch_size, seq_len)

            for b_states, b_actions, b_dones, (b_h0, b_c0), b_indices in data_loader:

                b_adv_list, b_ret_list, b_old_logp_list = [], [], []
                for idx in b_indices:
                    sl = slice(idx, idx + seq_len)
                    b_adv_list.append(advantages[sl])
                    b_ret_list.append(returns[sl])
                    b_old_logp_list.append(old_log_probs[sl])

                # Stack 后形状: [Batch, Seq]
                b_adv = torch.stack(b_adv_list)
                b_ret = torch.stack(b_ret_list)
                b_old_logp = torch.stack(b_old_logp_list)

                # 获取新分布
                # log_probs 形状: [Batch, Seq]
                dist, log_probs = self.net.get_dist_logp(b_states, (b_h0, b_c0), b_actions, terminated=b_dones)

                # Value 预测: [Batch, Seq, 1] -> [Batch, Seq]
                values_pred = self.net.get_value(b_states, (b_h0, b_c0), terminated=b_dones).squeeze(-1)

                # 确保维度匹配
                if log_probs.shape != b_old_logp.shape:
                    raise RuntimeError(f"Shape mismatch: new {log_probs.shape} vs old {b_old_logp.shape}")

                entropy_loss = dist.entropy().mean()

                ratio = torch.exp(log_probs - b_old_logp)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef) * b_adv
                actor_loss = -torch.mean(torch.min(surr1, surr2))

                critic_loss = 0.5 * ((values_pred - b_ret) ** 2).mean()

                loss = actor_loss + self.config.vf_coef * critic_loss - self.config.ent_coef * entropy_loss

                self.net.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.max_grad_norm)
                self.net.opt.step()

        self.buffer.reset()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = Config()  # 使用默认配置

    # 建立环境
    env = gym.make("CartPole-v1")
    # env = gym.make("LunarLander-v3") # 也可以试试更难的

    ac = RecurrentPPO(
        lr=3e-4,
        obs_dim=env.observation_space.shape[0],
        h_dim=128,
        action_dim=env.action_space.n,
        lstm_hidden_dim=config.lstm_hidden_dim,
        lstm_layers=config.lstm_layers,
        computes_grad=True,
        device=device
    )

    agent = PPOAgentLSTM('test', ac, config)

    reward_container = []
    avg_rewards = []
    best_avg = -float('inf')

    # 训练循环
    max_episodes = 2000
    pbar = tqdm(range(max_episodes))

    for i in pbar:
        state, _ = env.reset()
        episode_reward = 0
        done = False

        # 收集数据直到 Buffer 满 (模拟 CleanRL 的 num_steps)
        # 或者按 Episode 收集。这里保留你的风格：按 Episode 跑，但 step() 内检查是否该更新
        while not done:
            action = agent.action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # PPO 不需要这种人工 Reward Shaping，CartPole 原始 Reward 即可
            # 但如果你想保留：
            x, _, theta, _ = next_state
            r1 = (env.unwrapped.x_threshold - abs(x)) / env.unwrapped.x_threshold - 0.8
            r2 = (env.unwrapped.theta_threshold_radians - abs(theta)) / env.unwrapped.theta_threshold_radians - 0.5
            reward = 2 * r1 + r2

            agent.store(state, action, reward, terminated, truncated)

            state = next_state
            episode_reward += reward
            done = terminated or truncated

            # 如果 Buffer 满了，立即更新 (PPO 标准做法是收集固定步数)
            if len(agent.buffer) >= config.capacity:
                agent.step(batch_size=64)  # 这里的 batch_size 是 LSTM 序列数

        reward_container.append(episode_reward)
        avg_reward = np.mean(reward_container[-10:])
        avg_rewards.append(avg_reward)

        if avg_reward > best_avg:
            best_avg = avg_reward
            # agent.save()

        pbar.set_description(f"Ep Reward: {episode_reward:.1f}, Avg: {avg_reward:.1f}")

    env.close()

    # Plot results
    plt.plot(reward_container, alpha=0.5, label='Raw')
    plt.plot(avg_rewards, label='Avg 10')
    plt.legend()
    plt.show()
