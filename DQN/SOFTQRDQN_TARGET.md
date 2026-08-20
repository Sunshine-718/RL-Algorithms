# SoftQRDQN 分位数目标说明

这份文档记录 SoftQRDQN 的分位数 target 为什么必须使用一套统一的
动作概率，以及旧实现为什么会产生不自洽的回报分布。

## 必须保持的原则

环境中的执行顺序是：

1. Agent 根据当前状态选择一次动作；
2. 环境执行这个动作；
3. 该动作产生随机回报。

因此，策略只能为一个状态提供一套动作概率。一个动作被选中后，它的
完整回报分布必须整体保留。低分位点和高分位点不能重新选择不同动作。

## 网络输出的含义

假设网络输出：

```text
quantiles.shape = [B, A, N]
```

- `B`：batch size；
- `A`：动作数量；
- `N`：每个动作的分位点数量，Breakout 使用 51。

动作价值由该动作所有分位点的均值给出：

```python
q_values = quantiles.mean(dim=-1)  # [B, A]
```

Soft 策略也必须从这些动作均值统一计算：

```python
log_probabilities = torch.log_softmax(q_values / alpha, dim=1)
probabilities = log_probabilities.exp()
```

## 旧实现的问题

旧实现的核心形式是：

```python
next_quantiles = torch.minimum(
    online_quantiles,
    target_quantiles,
)

soft_quantiles = alpha * torch.logsumexp(
    next_quantiles / alpha,
    dim=1,
)
```

`dim=1` 是动作维。这段代码会为每个分位点分别在动作之间执行一次
`logsumexp`：

```text
第 1 个分位点使用一套动作偏好
第 2 个分位点再使用一套动作偏好
...
第 N 个分位点再使用一套动作偏好
```

当 `alpha` 较小时，`logsumexp` 接近 `max`。旧实现相当于允许低分位点
选择一个动作、高分位点选择另一个动作。真实 Agent 在执行动作前并不知道
未来回报会落在哪个分位区间，因此无法使用这种策略。

## 两动作示例

假设两个动作的回报分布如下：

| 动作 | 低分位点 | 高分位点 | 均值 |
|---|---:|---:|---:|
| A | 0 | 100 | 50 |
| B | 50 | 50 | 50 |

两个动作的均值相同，所以策略应当各以 50% 的概率选择它们。

真实混合分布为：

```text
25% 得到 0
50% 得到 50
25% 得到 100
```

忽略熵奖励时，均值仍然是 50。

旧实现会分别选择：

```text
低分位点：max(0, 50) = 50，来自动作 B
高分位点：max(100, 50) = 100，来自动作 A
```

最终得到 `[50, 100]`，均值变成 75。它等价于 Agent 在知道未来结果
属于低分位还是高分位后，再回头选择动作。

## 正确的分布式 Soft target

先使用 online 网络的动作均值计算唯一策略：

```python
online_quantiles = online_net(next_state)       # [B, A, N]
target_quantiles = target_net(next_state)       # [B, A, N]

log_probabilities = torch.log_softmax(
    online_quantiles.mean(dim=-1) / alpha,
    dim=1,
)
probabilities = log_probabilities.exp()
```

online 网络负责决定动作概率，target 网络负责提供每个动作的完整回报分布。

然后构造每个动作的 Soft return atoms：

```python
soft_target_atoms = (
    target_quantiles
    - alpha * log_probabilities.unsqueeze(-1)
)

target_atoms = (
    reward.unsqueeze(-1)
    + discount_n.unsqueeze(-1)
    * soft_target_atoms
    * (1.0 - terminated).unsqueeze(-1)
)
```

每个动作内的 `N` 个 atoms 均分该动作的概率：

```python
atom_weights = (
    probabilities.unsqueeze(-1).expand_as(target_quantiles) / N
)
```

对于 Breakout：

```text
A = 4
N = 51
target_atoms.shape = [B, 204]
atom_weights.shape = [B, 204]
```

每一行权重必须满足：

```python
atom_weights.sum(dim=1) == 1
```

Quantile Huber loss 在 target atom 维度按 `atom_weights` 加权求和，在
batch 和预测分位点维度求平均。

## 为什么移除逐元素 minimum

online 网络和 target 网络是同一网络在不同时间的参数。逐分位点取最小值
会把两个时刻的预测拼接起来：

```text
online: [8, 14]
target: [10, 12]
minimum: [8, 12]
```

结果中的低分位点来自 online，高分位点来自 target。上涨时旧 target 会
阻挡新估计，下跌时 online 的较低值会立即进入目标。

Clipped Double Q 需要两个独立训练的 critic。只有一套 online 网络和它的
滞后 target 副本时，不应逐分位点取 `minimum`。

## 防止回归的检查项

修改 SoftQRDQN target 时需要同时满足：

- 策略概率只从 `online_quantiles.mean(dim=-1)` 计算一次；
- 所有分位点共享同一套动作概率；
- target 网络提供完整的 `[B, A, N]` 回报分布；
- 每个 atom 的权重为 `pi(a|s) / N`；
- 每行 atom 权重之和为 1；
- terminal transition 的全部 target atoms 都等于即时回报；
- target 和 next-state online 分支不参与反向传播；
- 不对动作维执行逐分位点 `logsumexp`；
- 不对 online 和 target 副本逐分位点取 `minimum`。

对应回归测试位于：

```text
tests/test_softqrdqn_breakout.py
```

## 参考资料

- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
- [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)

