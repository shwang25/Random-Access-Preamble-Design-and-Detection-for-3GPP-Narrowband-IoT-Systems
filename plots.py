"""Generate plots from the saved NPRACH simulation results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import (
    DEFAULT_SIMULATION_MODE,
    ensure_results_dir,
    incompatible_run_configuration_keys,
    result_path,
)

os.environ.setdefault("MPLCONFIGDIR", str(ensure_results_dir() / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_SIMULATION_MODE,
        choices=("waveform", "fast"),
    )
    parser.add_argument(
        "--false-alarm-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--detection-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--toa-npz",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--toa-summary-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--toa-output",
        type=str,
        default=None,
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_npz_metadata(path: str | Path) -> dict:
    """Load the stored metadata from a ToA CDF NPZ bundle."""
    with np.load(path) as arrays:
        if "_metadata_json" not in arrays.files:
            raise ValueError(
                f"Missing `_metadata_json` in ToA bundle at {path}. "
                "Regenerate the ToA CDF results with the updated scripts."
            )
        metadata_json = str(arrays["_metadata_json"].item())
    return json.loads(metadata_json)


def require_run_configuration(metadata: dict, path: str | Path, expected_kind: str) -> dict:
    """Validate that a metadata blob has the expected kind and run configuration."""
    if metadata.get("result_kind") != expected_kind:
        raise ValueError(
            f"Expected `{expected_kind}` metadata in {path}, found `{metadata.get('result_kind')}`."
        )
    run_config = metadata.get("run_config")
    if not isinstance(run_config, dict):
        raise ValueError(f"Missing `run_config` metadata in {path}.")
    return run_config


def validate_result_bundle(
    mode: str,
    false_alarm_json: str | Path,
    detection_json: str | Path,
    toa_summary_json: str | Path,
    toa_npz: str | Path,
) -> None:
    """Ensure that the plot inputs come from one compatible result bundle."""
    false_alarm_metadata = load_json(false_alarm_json)
    detection_metadata = load_json(detection_json)
    toa_summary_metadata = load_json(toa_summary_json)
    toa_npz_metadata = load_npz_metadata(toa_npz)

    sources = [
        (
            "false alarm",
            false_alarm_json,
            require_run_configuration(false_alarm_metadata, false_alarm_json, "false_alarm"),
        ),
        (
            "detection",
            detection_json,
            require_run_configuration(detection_metadata, detection_json, "detection"),
        ),
        (
            "ToA summary",
            toa_summary_json,
            require_run_configuration(toa_summary_metadata, toa_summary_json, "toa_cdf_summary"),
        ),
        (
            "ToA samples",
            toa_npz,
            require_run_configuration(toa_npz_metadata, toa_npz, "toa_cdf_samples"),
        ),
    ]

    reference_label, reference_path, reference_run_config = sources[0]
    if reference_run_config.get("mode") != mode:
        raise ValueError(
            f"Requested `--mode {mode}` but {reference_label} metadata in {reference_path} "
            f"reports mode `{reference_run_config.get('mode')}`."
        )

    for label, path, run_config in sources[1:]:
        mismatched_keys = incompatible_run_configuration_keys(reference_run_config, run_config)
        if mismatched_keys:
            mismatch_string = ", ".join(mismatched_keys)
            raise ValueError(
                f"Incompatible plot inputs: {path} does not match {reference_path} on "
                f"{mismatch_string}."
            )


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
    fig, ax = plt.subplots(figsize=(8, 4.5))

    with np.load(toa_npz) as arrays:
        for key in sorted(key for key in arrays.files if key != "_metadata_json"):
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
    false_alarm_json = Path(args.false_alarm_json) if args.false_alarm_json else result_path(
        args.mode,
        "false_alarm_results.json",
    )
    detection_json = Path(args.detection_json) if args.detection_json else result_path(
        args.mode,
        "detection_results.json",
    )
    toa_npz = Path(args.toa_npz) if args.toa_npz else result_path(
        args.mode,
        "toa_cdf_results.npz",
    )
    toa_summary_json = (
        Path(args.toa_summary_json)
        if args.toa_summary_json
        else result_path(args.mode, "toa_cdf_summary.json")
    )
    summary_output = Path(args.summary_output) if args.summary_output else result_path(
        args.mode,
        "detection_false_alarm.png",
    )
    toa_output = Path(args.toa_output) if args.toa_output else result_path(
        args.mode,
        "toa_cdf.png",
    )

    validate_result_bundle(
        mode=args.mode,
        false_alarm_json=false_alarm_json,
        detection_json=detection_json,
        toa_summary_json=toa_summary_json,
        toa_npz=toa_npz,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    toa_output.parent.mkdir(parents=True, exist_ok=True)
    plot_detection_and_false_alarm(
        false_alarm_json=false_alarm_json,
        detection_json=detection_json,
        output_path=summary_output,
    )
    plot_toa_cdf(toa_npz=toa_npz, output_path=toa_output)
    print(f"Saved detection / false alarm plot to {summary_output}")
    print(f"Saved ToA CDF plot to {toa_output}")

    if args.show:
        img = plt.imread(summary_output)
        plt.figure(figsize=(8, 4.5))
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
