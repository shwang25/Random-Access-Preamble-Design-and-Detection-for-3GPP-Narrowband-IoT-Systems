"""Generate plots from the saved NPRACH simulation results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import ensure_results_dir

os.environ.setdefault("MPLCONFIGDIR", str(ensure_results_dir() / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    results_dir = ensure_results_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--false-alarm-json",
        type=str,
        default=str(results_dir / "false_alarm_results.json"),
    )
    parser.add_argument(
        "--detection-json",
        type=str,
        default=str(results_dir / "detection_results.json"),
    )
    parser.add_argument(
        "--toa-npz",
        type=str,
        default=str(results_dir / "toa_cdf_results.npz"),
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=str(results_dir / "detection_false_alarm.png"),
    )
    parser.add_argument(
        "--toa-output",
        type=str,
        default=str(results_dir / "toa_cdf.png"),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def empirical_cdf(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the x/y values of the empirical CDF."""
    samples = np.sort(np.asarray(samples, dtype=float))
    probabilities = np.arange(1, samples.size + 1, dtype=float) / max(samples.size, 1)
    return samples, probabilities


def plot_detection_and_false_alarm(
    false_alarm_json: str | Path,
    detection_json: str | Path,
    output_path: str | Path,
) -> None:
    """Plot misdetection and false alarm probabilities."""
    false_alarm = load_json(false_alarm_json)["cases"]
    detection = load_json(detection_json)["cases"]

    lengths = [entry["L"] for entry in detection]
    pmd = [entry["misdetection_probability"] for entry in detection]
    pfa = [entry["achieved_pfa"] for entry in false_alarm]

    x = np.arange(len(lengths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2.0, pmd, width=width, label="Misdetection")
    ax.bar(x + width / 2.0, pfa, width=width, label="False alarm")
    ax.set_xticks(x)
    ax.set_xticklabels([str(length) for length in lengths])
    ax.set_xlabel("Preamble length L (symbol groups)")
    ax.set_ylabel("Probability")
    ax.set_title("NPRACH Detection and False Alarm")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_toa_cdf(toa_npz: str | Path, output_path: str | Path) -> None:
    """Plot the empirical ToA error CDF."""
    arrays = np.load(toa_npz)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for key in sorted(arrays.files):
        errors = arrays[key]
        x, y = empirical_cdf(errors)
        label = key.replace("errors_L_", "") + " symbol groups"
        ax.step(x, y, where="post", label=label)

    ax.set_xlabel("Time-of-arrival estimation error (us)")
    ax.set_ylabel("CDF")
    ax.set_title("NPRACH ToA Estimation Error")
    ax.set_xlim(-3.0, 3.0)
    ax.set_xticks(np.arange(-3.0, 4.0, 1.0))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Entry point for plotting."""
    args = parse_args()
    plot_detection_and_false_alarm(
        false_alarm_json=args.false_alarm_json,
        detection_json=args.detection_json,
        output_path=args.summary_output,
    )
    plot_toa_cdf(toa_npz=args.toa_npz, output_path=args.toa_output)
    print(f"Saved detection / false alarm plot to {args.summary_output}")
    print(f"Saved ToA CDF plot to {args.toa_output}")

    if args.show:
        img = plt.imread(args.summary_output)
        plt.figure(figsize=(8, 4.5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
