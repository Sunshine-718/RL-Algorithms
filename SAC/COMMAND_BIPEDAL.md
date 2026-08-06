# 可键盘控制的 BipedalWalker

这个环境把按键转换为目标水平速度，QRSAC 输出四个关节动作。当前只包含左右移动和松键站立，没有跳跃。

## 控制与观测

- `A` / `←`：目标速度 `-command_speed`
- `D` / `→`：目标速度 `+command_speed`
- 松开按键或同时按左右键：目标速度 `0`
- `R`：重置环境
- `Esc`：退出

原始按键目标允许瞬间反向。环境内部使用二阶参考模型生成连续的参考速度与参考加速度，并限制最大加速度和 jerk。策略会同时看到原始目标、参考速度、参考加速度、实测加速度和相对高度，因此不需要历史帧也能区分“刚收到反向指令”和“已稳定匀速”。默认观测为 39 维：原始 24 维、向左的 10 条激光、5 个控制特征。

## 奖励

移动时同时优化：

- 水平速度对参考速度的误差；
- 实测加速度对参考加速度的误差；
- 躯干角度、角速度、竖直速度、动作幅度和动作变化。

参考加速度较大时，加速度跟踪权重自动提高；匀速阶段参考加速度逐渐回到零，速度跟踪成为主要目标。

当参考速度和参考加速度都接近零时，才启用站立奖励。站立高度奖励要求双脚接触地面，并在 `standing_height` 处封顶；同时奖励低水平速度、低竖直速度和低角速度。这样可以鼓励高而稳定的站姿，并限制跳起刷高度。

`acceleration_limit` 是参考轨迹的训练参数，不等同于环境给出的固定物理上限。默认 `3.0` 只作为初始值。训练后应根据 `evaluations.csv` 的加速度误差和成功轨迹调整：持续跟不上时降低该值，长期轻松跟随时再逐步提高。`jerk_limit` 还会限制加速度变化速度，避免目标加速度本身瞬间翻转。

## 训练

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

默认使用 8 个异步环境：

```bash
python3 SAC/train_command_bipedal_vector.py \
  --run-dir runs/qrsac_command_bipedal \
  --num-envs 8 \
  --vector-mode async
```

每个环境单独保存当前 episode。episode 完成后再写入 replay buffer，防止 n-step 回报跨环境或跨 episode。Gymnasium 的 `NEXT_STEP` 自动重置产生的空重置步不会进入 replay buffer。

短流程检查：

```bash
python3 SAC/train_command_bipedal_vector.py \
  --run-dir runs/qrsac_command_smoke \
  --smoke-test
```

恢复模型和优化器状态：

```bash
python3 SAC/train_command_bipedal_vector.py \
  --run-dir runs/qrsac_command_bipedal \
  --resume runs/qrsac_command_bipedal/latest.pt
```

Replay buffer 不写入检查点，恢复后会重新收集样本，网络更新会等到 buffer 再次达到 batch 大小。

## 查看与操作

查看训练状态：

```bash
python3 SAC/monitor_command_bipedal.py runs/qrsac_command_bipedal
```

加载模型并使用键盘：

```bash
python3 SAC/play_command_bipedal.py \
  runs/qrsac_command_bipedal/latest.pt
```

主要产物：

- `status.json`：当前步数、episode、buffer 和最近损失；
- `episodes.csv`：每个 episode 的回报、速度 RMSE、加速度 RMSE、站立高度；
- `evaluations.csv`：固定循环指令下的确定性评估；
- `metrics.jsonl`：episode 与评估事件；
- `latest.pt`：模型、目标网络、优化器、训练计数和环境配置。
