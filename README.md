# RL-Algorithms

一个简洁的强化学习算法实现库，包含多种主流算法及其变体。

## 包含算法

### 1. DQN (Deep Q-Network) 系列
- DQN (Deep Q-Network)
- IQN (Implicit Quantile Networks)
- QR-DQN (Quantile Regression DQN)
- Soft DQN variants
- SPR-DQN

### 2. PPO (Proximal Policy Optimization) 系列
- PPO (离散动作空间)
- PPO (连续动作空间)
- PPO with LSTM
- QR-PPO variants

### 3. 其他算法
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

## 项目结构

```
RL-Algorithms/
├── DDPG/          # DDPG算法实现
├── DQN/           # DQN系列算法
├── PPO/           # PPO系列算法
├── SAC/           # SAC算法
├── TD3/           # TD3算法
├── .gitignore
└── README.md
```

每个算法文件夹通常包含：
- 算法主文件 (如 dqn.py, ppo_discrete.py)
- 通用工具模块 (common.py)
- 参数配置 (params/)
- 其他相关模块

## 使用方法

1. 克隆仓库：
```bash
git clone <repository-url>
cd RL-Algorithms
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行特定算法：
```python
# 示例：运行DQN
python DQN/dqn.py
```

## 依赖要求

主要依赖：
- Python 3.7+
- PyTorch
- Gym / Gymnasium
- NumPy

具体依赖请查看各算法文件夹内的requirements.txt或导入语句。

## 注意事项

- 每个算法文件夹是相对独立的，可以根据需要单独使用
- 参数配置通常在params/文件夹中
- 部分算法可能需要特定版本的依赖库

## 致谢

本项目参考了多个开源强化学习实现，感谢相关作者的工作。

---

*简单、清晰的强化学习算法实现，适合学习和研究使用。*