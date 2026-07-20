"""
Export all TensorBoard scalar data (from every run under a logdir) to CSV files.

Usage:
    python export_tb_to_csv.py output/tensorboard/ --out tb_csv/

Produces one CSV per (run, tag) pair, e.g.:
    tb_csv/PPO_1__rollout_ep_rew_mean.csv
    tb_csv/PPO_1__train_mean_abs_action.csv
    tb_csv/PPO_2__rollout_ep_rew_mean.csv
    ...
Each CSV has columns: step, wall_time, value

After running this, zip the output folder — it'll be tiny compared to raw
tfevents files, and you can upload/paste those CSVs directly for analysis.
"""

import argparse
import csv
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_run(run_dir: Path, out_dir: Path) -> None:
    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        return

    run_name = run_dir.name
    for tag in tags:
        events = ea.Scalars(tag)
        safe_tag = tag.replace("/", "_")
        out_path = out_dir / f"{run_name}__{safe_tag}.csv"

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wall_time", "value"])
            for e in events:
                writer.writerow([e.step, e.wall_time, e.value])

        print(f"Wrote {out_path}  ({len(events)} points)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Path to tensorboard logdir (contains PPO_1/, PPO_2/, etc.)")
    parser.add_argument("--out", type=str, default="tb_csv", help="Output folder for CSVs")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [d for d in logdir.iterdir() if d.is_dir()]
    if not run_dirs:
        print(f"No run subfolders found under {logdir}")
        return

    for run_dir in sorted(run_dirs):
        export_run(run_dir, out_dir)

    print(f"\nDone. CSVs written to {out_dir}/ — zip that folder to share or upload.")


if __name__ == "__main__":
    main()
