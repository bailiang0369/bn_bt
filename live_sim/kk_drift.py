"""h10 全特征漂移诊断的 Kaggle 启动器（新阶段独立文件）。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import kaggle_launcher as L  # noqa: E402

L.PROJECTS = {
    "drift": dict(
        slug="byh850515/h10-feature-drift-diag",
        title="H10 Feature Drift Diag",
        entry="live_sim.h10_drift_diag",
        enable_internet=True,
        dataset_sources=[],
        env={"PRE_DAYS": "30", "WIN_DAYS": "1460", "N_PERM": "5000"},
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="h10 特征漂移诊断启动器")
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
    L.run("drift", push=not a.no_push, wait=not a.no_wait,
          logs=not a.no_logs, parse=True, overrides=overrides)


if __name__ == "__main__":
    main()
