# QRSAC + PD 控制 BipedalWalker

这个实验采用两层控制：QRSAC 输出 4 个归一化关节目标，PD 控制器把目标角转换为环境的 4 维电机指令。

```text
24 维状态 → QRSAC → 关节目标角 → PD → 电机方向与最大力矩比例
```

## 控制公式

策略动作会按 BipedalWalker 自身的关节限位映射：

- hip：`[-0.8, 1.1] rad`
- knee：`[-1.6, -0.1] rad`

每个关节的控制量为：

```text
u = clip(Kp * (q_target - q) - Kd * q_dot, -1, 1)
```

D 项使用测量角速度。这样不会因 QRSAC 改变目标角而产生微分冲击，也不需要给 observation 增加 PID 历史状态。经验回放保存 QRSAC 的目标动作，环境奖励中的能耗惩罚使用 PD 的实际电机输出。

默认增益：

```text
Kp = [2.00, 2.00, 2.00, 2.00]
Kd = [0.15, 0.10, 0.15, 0.10]
```

四项依次对应左 hip、左 knee、右 hip、右 knee。

## 运行

短时检查：

```bash
python3 SAC/qrsac_pd_bipedalwalker.py \
  --smoke-test \
  --vector-mode sync \
  --run-dir runs/qrsac_pd_smoke
```

正式训练默认持续运行，按 `Ctrl+C` 时保存 `last.pt`：

```bash
python3 SAC/qrsac_pd_bipedalwalker.py
```

恢复训练：

```bash
python3 SAC/qrsac_pd_bipedalwalker.py --resume
```

无界面评估：

```bash
python3 SAC/qrsac_pd_bipedalwalker.py --evaluate --no-render
```

单独调整 hip 与 knee 增益：

```bash
python3 SAC/qrsac_pd_bipedalwalker.py \
  --kp 2.5 2.0 2.5 2.0 \
  --kd 0.18 0.12 0.18 0.12
```

不同 PD 增益会改变策略动作的实际含义。更换增益后应使用新的 `--run-dir`，不要恢复旧 checkpoint。
