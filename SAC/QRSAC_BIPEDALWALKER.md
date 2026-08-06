# BipedalWalker QRSAC

The trainer is headless and writes all progress to a run directory. From the
repository root, start a foreground run with:

```powershell
python SAC\train_qrsac_bipedalwalker.py --run-dir runs\qrsac_bipedalwalker --device cuda
```

Inspect the most recent completed episode without attaching to the trainer:

```powershell
python SAC\monitor_qrsac_bipedalwalker.py --run-dir runs\qrsac_bipedalwalker
```

Run files:

- `status.json`: current state, step, returns, losses, throughput, and ETA
- `episodes.csv`: one row per training episode
- `metrics.jsonl`: episode and deterministic evaluation events
- `latest.pt`: resumable periodic checkpoint
- `best.pt`: checkpoint with the highest deterministic evaluation mean
- `pid.txt`: background trainer process ID

Resume a stopped run:

```powershell
python SAC\train_qrsac_bipedalwalker.py `
  --run-dir runs\qrsac_bipedalwalker `
  --resume runs\qrsac_bipedalwalker\latest.pt `
  --device cuda
```

To request a graceful stop, create an empty `STOP` file in the run directory.
The trainer finishes the current safe point and refreshes `latest.pt` before
exiting. Remove the file before resuming.

Evaluate a checkpoint without training:

```powershell
python SAC\train_qrsac_bipedalwalker.py `
  --eval-only `
  --resume runs\qrsac_bipedalwalker\best.pt `
  --run-dir runs\qrsac_bipedalwalker_eval `
  --eval-episodes 10 `
  --device cuda
```

Fine-tune the normal-walker weights in the hardcore environment with a fresh
replay buffer, fresh optimizers, and reset entropy temperature:

```powershell
python SAC\train_qrsac_bipedalwalker.py `
  --hardcore `
  --init-from runs\qrsac_bipedalwalker\best.pt `
  --run-dir runs\qrsac_bipedalwalker_hardcore `
  --device cuda
```

Vectorized collection keeps one complete episode cache per environment and
flushes those caches sequentially through the unchanged replay buffer:

```powershell
python SAC\train_qrsac_bipedalwalker_vector.py `
  --num-envs 8 `
  --vector-mode async `
  --run-dir runs\qrsac_bipedalwalker_vector `
  --device cuda
```

`--gradient-steps-per-vector-step 1` consumes one gradient batch for every
`num-envs` collected transitions. Set it equal to `num-envs` to retain an
update-to-data ratio close to the single-environment trainer.
