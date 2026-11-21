"""新阶段 h3 工作日阈值对照的专用 Kaggle 启动器（不改旧 launcher）。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kaggle_launcher as L  # noqa: E402

L.PROJECTS = {
    "h3wd": dict(
        slug="byh850515/h3-weekday-threshold-replay",
        title="H3 Weekday Threshold Replay",
        entry="live_sim.h3_weekday_sim",
        enable_internet=True,
        dataset_sources=[],
        env={"TPD": "15", "TPD_WD": "15", "PAYOUT": "0.80",
             "SLOTS": "3", "OFFSET_MIN": "0", "BOOT": "100", "PLACEBO": "50"},
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="h3 工作日阈值 replay 启动器")
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
    L.run("h3wd", push=not a.no_push, wait=not a.no_wait,
          logs=not a.no_logs, parse=True, overrides=overrides)


if __name__ == "__main__":
    main()
