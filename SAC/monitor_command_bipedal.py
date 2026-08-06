"""Read the atomic status file produced by command BipedalWalker training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def render_status(status: dict) -> str:
    return (
        f"step={status.get('global_step', 0)} "
        f"episodes={status.get('completed_episodes', 0)} "
        f"buffer={status.get('buffer_size', 0)} "
        f"return100={status.get('recent_return_mean', 0.0):.3f} "
        f"critic={status.get('critic_loss', 0.0):.4f} "
        f"actor={status.get('actor_loss', 0.0):.4f} "
        f"alpha={status.get('alpha', 0.0):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor command training status.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    status_path = args.run_dir.resolve() / "status.json"
    try:
        while True:
            if status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                print(render_status(status), flush=True)
                if status.get("completed") or args.once:
                    break
            elif args.once:
                raise FileNotFoundError(status_path)
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
