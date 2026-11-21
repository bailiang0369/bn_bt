"""新阶段专用 Kaggle 启动器：不修改旧 kaggle_launcher.py，只注入新项目配置。

用法：
  python live_sim/kk.py                          # 完整跑（默认配置）
  python live_sim/kk.py --set ROUNDS=100 --set SEEDS=42,43 --set START_MONTH=2026-10 \
      --set END_MONTH=2026-10 --set PERTURB_N=4  # 先跑 smoke
  python live_sim/kk.py --no-wait --no-logs      # 只推
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kaggle_launcher as L  # noqa: E402

L.PROJECTS = {
    "livesim": dict(
        slug="byh850515/h10-live-replay-sim-new-stage",
        title="H10 Live Replay Sim New Stage",
        entry="live_sim.h10_live_sim",
        enable_internet=True,
        dataset_sources=[],
        env={"PERTURB_N": "12", "OFFSET_MIN": "0", "SLOTS": "3",
             "DEDUP": "none", "COOLDOWN_MIN": "10"},
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="live_sim 专用 Kaggle 启动器")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="K=V")
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        if "=" not in kv:
            ap.error(f"--set 需要 K=V，收到 {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = v.strip()
    L.run("livesim", push=not a.no_push, wait=not a.no_wait,
          logs=not a.no_logs, parse=True, overrides=overrides)


if __name__ == "__main__":
    main()
