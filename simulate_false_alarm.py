"""Run the false alarm experiment and threshold calibration for the paper cases."""

from __future__ import annotations

import argparse
import json

from config import (
    DEFAULT_PAPER_CHANNEL_MODEL,
    DEFAULT_SEARCH_M1,
    DEFAULT_SEARCH_M2,
    DEFAULT_SEED,
    DEFAULT_SIMULATION_MODE,
    PAPER_ITERATIONS_FALSE_ALARM,
    PAPER_NUM_RX,
    build_search_grid,
    derive_parameters,
    ensure_results_dir,
    make_rng,
    paper_coverage_cases,
)
from detector import run_false_alarm_experiment
from waveform import build_preamble_reference_bank, design_front_end_filter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-iterations", type=int, default=PAPER_ITERATIONS_FALSE_ALARM)
    parser.add_argument("--evaluation-iterations", type=int, default=PAPER_ITERATIONS_FALSE_ALARM)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--m1", type=int, default=DEFAULT_SEARCH_M1)
    parser.add_argument("--m2", type=int, default=DEFAULT_SEARCH_M2)
    parser.add_argument("--num-rx", type=int, default=PAPER_NUM_RX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mode", type=str, default=DEFAULT_SIMULATION_MODE, choices=("waveform", "fast"))
    parser.add_argument("--channel-model", type=str, default=DEFAULT_PAPER_CHANNEL_MODEL)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ensure_results_dir() / "false_alarm_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the false alarm simulation."""
    args = parse_args()
    rng = make_rng(args.seed)
    output = {"mode": args.mode, "channel_model": args.channel_model, "cases": []}

    for case in paper_coverage_cases():
        params = derive_parameters(case.preamble_length)
        search_grid = build_search_grid(params, M1=args.m1, M2=args.m2)
        front_end = design_front_end_filter(params)
        references = build_preamble_reference_bank(
            params=params,
            front_end=front_end,
            hopping_seed=args.seed,
        )
        result = run_false_alarm_experiment(
            params=params,
            references=references,
            search_grid=search_grid,
            snr_db=case.snr_db,
            rng=rng,
            calibration_trials=args.calibration_iterations,
            evaluation_trials=args.evaluation_iterations,
            batch_size=args.batch_size,
            num_rx=args.num_rx,
            mode=args.mode,
            front_end=front_end,
        )

        record = {
            "L": case.preamble_length,
            "snr_db": case.snr_db,
            "mode": args.mode,
            "channel_model": args.channel_model,
            "threshold": result["threshold"],
            "achieved_pfa": result["achieved_pfa"],
            "target_pfa": result["target_pfa"],
            "calibration_trials": result["calibration_trials"],
            "evaluation_trials": result["evaluation_trials"],
            "calibration_statistics_mean": result["calibration_statistics_mean"],
            "calibration_statistics_std": result["calibration_statistics_std"],
            "evaluation_statistics_mean": result["evaluation_statistics_mean"],
            "evaluation_statistics_std": result["evaluation_statistics_std"],
            "calibration_pfa": result["calibration_pfa"],
            "evaluation_false_alarm_count": result["evaluation_false_alarm_count"],
        }
        output["cases"].append(record)

        print(
            f"[False alarm] L={case.preamble_length:3d} "
            f"mode={args.mode:8s} "
            f"SNR={case.snr_db:6.2f} dB "
            f"threshold={result['threshold']:.4f} "
            f"cal_Pfa={result['calibration_pfa']:.6f} "
            f"eval_Pfa={result['achieved_pfa']:.6f}"
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"Saved false alarm results to {args.output}")


if __name__ == "__main__":
    main()
