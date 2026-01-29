import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam, SGD


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    # pred: [B, N], target: [B, 1], tau: [1, N]
    error = pred.unsqueeze(2) - target.expand_as(pred).unsqueeze(1)  # [B, N, N]
    huber = torch.where(error.abs() <= kappa, 0.5 * error.pow(2), kappa * (error.abs() - 0.5 * kappa))
    loss = torch.abs(tau.unsqueeze(-1) - (error.detach() < 0).float()) * huber  # [B, N, N]
    return loss.mean()


class NetworkBase(nn.Module):
    @staticmethod
    def configure_optimizer(model, weight_decay, learning_rate, betas=(0.9, 0.999)):
        decay_params = []
        nodecay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'norm' in name or name.endswith('.bias'):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        return NAdam(param_groups, lr=learning_rate, betas=betas, decoupled_weight_decay=True)

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


class AgentBase:
    @property
    def n_step(self):
        return self._n_step

    @n_step.setter
    def n_step(self, val):
        assert val >= 1 and isinstance(val, int)
        self._n_step = val
        self.buffer.n_step = val

    def cache(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.cache_transition(state, action, reward, next_state, terminated, truncated)

    def process(self):
        self.buffer.process()

    def save(self, model='last'):
        if self.params is not None:
            self.net.save(f'{self.params}/{self.name}_{model}.pt')
        else:
            self.net.save(f'{self.name}_{model}.pt')

    def load(self, model='last'):
        if self.params is not None:
            self.net.load(f'{self.params}/{self.name}_{model}.pt')
        else:
            self.net.load(f'{self.name}_{model}.pt')
        self.soft_update(tau=1)

    def decay_noise(self, zero_noise=False):
        self.noise = max(self.min_noise, self.noise * self.decay) * (1 - bool(zero_noise))

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
