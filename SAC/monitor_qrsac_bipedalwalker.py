"""Print a compact snapshot of a background QRSAC run."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, default=Path("runs/qrsac_bipedalwalker"))
args = parser.parse_args()
status_path = args.run_dir.resolve() / "status.json"
if not status_path.exists():
    raise SystemExit(f"No status file yet: {status_path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
print(json.dumps(status, indent=2, ensure_ascii=False))
