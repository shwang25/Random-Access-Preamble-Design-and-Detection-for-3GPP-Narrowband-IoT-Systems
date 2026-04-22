"""Detection logic and Monte Carlo experiment drivers for waveform-domain NPRACH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from channel import (
    apply_channel_to_waveform,
    generate_waveform_noise_only_input,
    noise_variance_from_snr_db,
)
from config import TARGET_PFA, DerivedParameters, SearchGrid, circular_delay_difference_samples
from receiver import fft_joint_search_batch, process_received_waveform_batch_to_symbols
from waveform import PreambleWaveformReference, WaveformFrontEnd


@dataclass(frozen=True)
class ThresholdCalibration:
    """Calibrated detection threshold and the associated empirical Pfa."""

    threshold: float
    achieved_pfa: float
    target_pfa: float
    statistics: np.ndarray


def estimate_waveform_receiver_noise_gain(
    params: DerivedParameters,
    reference: PreambleWaveformReference,
    front_end: WaveformFrontEnd,
    rng: np.random.Generator,
    num_realizations: int = 64,
) -> float:
    """
    Empirically map receiver-input white-noise variance to extracted-symbol variance.

    This keeps the waveform-domain SNR tied to the same extracted-symbol domain
    used by the Eq. (4)-(6) estimator.
    """
    rx_noise = generate_waveform_noise_only_input(
        params=params,
        front_end=front_end,
        input_noise_variance=1.0,
        rng=rng,
        num_rx=1,
        batch_size=num_realizations,
        extra_tail_samples=params.cp_samples,
    )
    _, extracted = process_received_waveform_batch_to_symbols(
        params=params,
        rx_waveform_oversampled=rx_noise,
        front_end=front_end,
        hop_sequence=reference.hop_sequence,
    )
    return float(np.mean(np.abs(extracted) ** 2))


def waveform_input_noise_variance_for_snr(
    params: DerivedParameters,
    reference: PreambleWaveformReference,
    front_end: WaveformFrontEnd,
    snr_db: float,
    rng: np.random.Generator,
) -> float:
    """Convert the target extracted-symbol SNR into receiver-input noise variance."""
    extracted_noise_variance = noise_variance_from_snr_db(snr_db)
    receiver_noise_gain = estimate_waveform_receiver_noise_gain(
        params=params,
        reference=reference,
        front_end=front_end,
        rng=rng,
    )
    return extracted_noise_variance / receiver_noise_gain


def _sample_true_toa_samples(
    params: DerivedParameters,
    rng: np.random.Generator,
    batch_size: int,
    front_end: WaveformFrontEnd,
) -> np.ndarray:
    """Sample timing uncertainty on the oversampled waveform grid."""
    true_delay_upsampled = rng.integers(
        low=0,
        high=params.cp_samples * front_end.oversampling_factor,
        size=batch_size,
    )
    return true_delay_upsampled.astype(float) / front_end.oversampling_factor


def _simulate_waveform_detection_subbatch(
    params: DerivedParameters,
    reference: PreambleWaveformReference,
    front_end: WaveformFrontEnd,
    toa_samples: np.ndarray,
    cfo_hz: np.ndarray,
    drift_hz_per_s: np.ndarray,
    input_noise_variance: float,
    rng: np.random.Generator,
    num_rx: int,
    channel_model: str,
    doppler_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one waveform-domain signal-present subbatch for one preamble."""
    oversampled_waveforms = []
    effective_true_toa = np.empty(toa_samples.size, dtype=float)

    for trial_index in range(toa_samples.size):
        realization = apply_channel_to_waveform(
            params=params,
            tx_waveform_upsampled=reference.tx_waveform_upsampled,
            front_end=front_end,
            toa_samples=float(toa_samples[trial_index]),
            cfo_hz=float(cfo_hz[trial_index]),
            drift_hz_per_s=float(drift_hz_per_s[trial_index]),
            input_noise_variance=input_noise_variance,
            rng=rng,
            num_rx=num_rx,
            channel_model=channel_model,
            doppler_hz=doppler_hz,
        )
        oversampled_waveforms.append(realization.rx_waveform_oversampled)
        effective_true_toa[trial_index] = realization.true_delay_samples

    max_length = max(waveform.shape[1] for waveform in oversampled_waveforms)
    rx_batch = np.zeros((toa_samples.size, num_rx, max_length), dtype=complex)
    for trial_index, waveform in enumerate(oversampled_waveforms):
        rx_batch[trial_index, :, : waveform.shape[1]] = waveform

    _, extracted = process_received_waveform_batch_to_symbols(
        params=params,
        rx_waveform_oversampled=rx_batch,
        front_end=front_end,
        hop_sequence=reference.hop_sequence,
    )
    return extracted, effective_true_toa


def _simulate_waveform_noise_only_subbatch(
    params: DerivedParameters,
    reference: PreambleWaveformReference,
    front_end: WaveformFrontEnd,
    batch_size: int,
    input_noise_variance: float,
    rng: np.random.Generator,
    num_rx: int,
) -> np.ndarray:
    """Simulate one waveform-domain false-alarm subbatch."""
    noise_input = generate_waveform_noise_only_input(
        params=params,
        front_end=front_end,
        input_noise_variance=input_noise_variance,
        rng=rng,
        num_rx=num_rx,
        batch_size=batch_size,
        extra_tail_samples=params.cp_samples,
    )
    _, extracted = process_received_waveform_batch_to_symbols(
        params=params,
        rx_waveform_oversampled=noise_input,
        front_end=front_end,
        hop_sequence=reference.hop_sequence,
    )
    return extracted


def collect_noise_only_statistics(
    params: DerivedParameters,
    references: dict[int, PreambleWaveformReference],
    search_grid: SearchGrid,
    snr_db: float,
    rng: np.random.Generator,
    num_trials: int,
    batch_size: int,
    num_rx: int,
    front_end: WaveformFrontEnd,
) -> np.ndarray:
    """Collect waveform-domain noise-only decision statistics."""
    statistics = np.empty(num_trials, dtype=float)
    cursor = 0

    reference_zero = references[0]
    waveform_noise_variance = waveform_input_noise_variance_for_snr(
        params=params,
        reference=reference_zero,
        front_end=front_end,
        snr_db=snr_db,
        rng=np.random.default_rng(314159),
    )

    while cursor < num_trials:
        current_batch = min(batch_size, num_trials - cursor)
        extracted = _simulate_waveform_noise_only_subbatch(
            params=params,
            reference=reference_zero,
            front_end=front_end,
            batch_size=current_batch,
            input_noise_variance=waveform_noise_variance,
            rng=rng,
            num_rx=num_rx,
        )
        result = fft_joint_search_batch(
            params=params,
            hop_sequence=reference_zero.hop_sequence,
            z_symbols=extracted,
            search_grid=search_grid,
            return_surface=False,
        )
        statistics[cursor : cursor + current_batch] = result["statistics"]
        cursor += current_batch
    return statistics


def calibrate_threshold(
    params: DerivedParameters,
    references: dict[int, PreambleWaveformReference],
    search_grid: SearchGrid,
    snr_db: float,
    rng: np.random.Generator,
    num_trials: int,
    batch_size: int,
    num_rx: int,
    front_end: WaveformFrontEnd,
    target_pfa: float = TARGET_PFA,
) -> ThresholdCalibration:
    """Calibrate the threshold from an independent noise-only data set."""
    statistics = collect_noise_only_statistics(
        params=params,
        references=references,
        search_grid=search_grid,
        snr_db=snr_db,
        rng=rng,
        num_trials=num_trials,
        batch_size=batch_size,
        num_rx=num_rx,
        front_end=front_end,
    )

    threshold = float(np.quantile(statistics, 1.0 - target_pfa, method="higher"))
    achieved_pfa = float(np.mean(statistics > threshold))
    return ThresholdCalibration(
        threshold=threshold,
        achieved_pfa=achieved_pfa,
        target_pfa=target_pfa,
        statistics=statistics,
    )


def run_false_alarm_experiment(
    params: DerivedParameters,
    references: dict[int, PreambleWaveformReference],
    search_grid: SearchGrid,
    snr_db: float,
    rng: np.random.Generator,
    calibration_trials: int,
    evaluation_trials: int,
    batch_size: int,
    num_rx: int,
    front_end: WaveformFrontEnd,
    threshold: float | None = None,
    target_pfa: float = TARGET_PFA,
) -> dict:
    """Calibrate and then evaluate false alarm probability on independent data."""
    calibration = calibrate_threshold(
        params=params,
        references=references,
        search_grid=search_grid,
        snr_db=snr_db,
        rng=rng,
        num_trials=calibration_trials,
        batch_size=batch_size,
        num_rx=num_rx,
        front_end=front_end,
        target_pfa=target_pfa,
    )
    used_threshold = calibration.threshold if threshold is None else float(threshold)
    evaluation_statistics = collect_noise_only_statistics(
        params=params,
        references=references,
        search_grid=search_grid,
        snr_db=snr_db,
        rng=rng,
        num_trials=evaluation_trials,
        batch_size=batch_size,
        num_rx=num_rx,
        front_end=front_end,
    )
    achieved_pfa = float(np.mean(evaluation_statistics > used_threshold))

    return {
        "threshold": float(used_threshold),
        "achieved_pfa": achieved_pfa,
        "target_pfa": float(target_pfa),
        "calibration_trials": int(calibration_trials),
        "evaluation_trials": int(evaluation_trials),
        "calibration_statistics_mean": float(np.mean(calibration.statistics)),
        "calibration_statistics_std": float(np.std(calibration.statistics)),
        "evaluation_statistics_mean": float(np.mean(evaluation_statistics)),
        "evaluation_statistics_std": float(np.std(evaluation_statistics)),
        "calibration_pfa": calibration.achieved_pfa,
        "evaluation_false_alarm_count": int(np.count_nonzero(evaluation_statistics > used_threshold)),
    }


def run_detection_experiment(
    params: DerivedParameters,
    references: dict[int, PreambleWaveformReference],
    search_grid: SearchGrid,
    snr_db: float,
    threshold: float,
    rng: np.random.Generator,
    num_trials: int,
    batch_size: int,
    num_rx: int,
    channel_model: str,
    doppler_hz: float,
    front_end: WaveformFrontEnd,
    collect_errors: bool = False,
) -> dict:
    """Run the waveform-domain preamble-present Monte Carlo experiment."""
    statistics = np.empty(num_trials, dtype=float)
    detected = np.zeros(num_trials, dtype=bool)
    cfo_estimates = np.empty(num_trials, dtype=float)
    toa_estimates = np.empty(num_trials, dtype=float)
    cfo_truth = np.empty(num_trials, dtype=float)
    toa_truth = np.empty(num_trials, dtype=float)

    waveform_noise_variance = waveform_input_noise_variance_for_snr(
        params=params,
        reference=references[0],
        front_end=front_end,
        snr_db=snr_db,
        rng=np.random.default_rng(271828),
    )

    toa_errors_us = []
    cursor = 0
    while cursor < num_trials:
        current_batch = min(batch_size, num_trials - cursor)
        true_toa = _sample_true_toa_samples(
            params=params,
            rng=rng,
            batch_size=current_batch,
            front_end=front_end,
        )
        true_cfo = rng.uniform(-50.0, 50.0, size=current_batch)
        true_drift = rng.uniform(-22.5, 22.5, size=current_batch)
        preamble_indices = rng.integers(0, params.band_subcarriers, size=current_batch)

        batch_statistics = np.empty(current_batch, dtype=float)
        batch_detected = np.zeros(current_batch, dtype=bool)
        batch_cfo_estimates = np.empty(current_batch, dtype=float)
        batch_toa_estimates = np.empty(current_batch, dtype=float)
        batch_toa_truth = np.empty(current_batch, dtype=float)

        for preamble_index in np.unique(preamble_indices):
            mask = preamble_indices == preamble_index
            reference = references[int(preamble_index)]
            observations, effective_true_toa = _simulate_waveform_detection_subbatch(
                params=params,
                reference=reference,
                front_end=front_end,
                toa_samples=true_toa[mask],
                cfo_hz=true_cfo[mask],
                drift_hz_per_s=true_drift[mask],
                input_noise_variance=waveform_noise_variance,
                rng=rng,
                num_rx=num_rx,
                channel_model=channel_model,
                doppler_hz=doppler_hz,
            )
            result = fft_joint_search_batch(
                params=params,
                hop_sequence=reference.hop_sequence,
                z_symbols=observations,
                search_grid=search_grid,
                return_surface=False,
            )
            batch_statistics[mask] = result["statistics"]
            batch_detected[mask] = result["statistics"] > threshold
            batch_cfo_estimates[mask] = result["cfo_hz"]
            batch_toa_estimates[mask] = result["toa_samples"]
            batch_toa_truth[mask] = effective_true_toa

        batch_slice = slice(cursor, cursor + current_batch)
        statistics[batch_slice] = batch_statistics
        detected[batch_slice] = batch_detected
        cfo_estimates[batch_slice] = batch_cfo_estimates
        toa_estimates[batch_slice] = batch_toa_estimates
        cfo_truth[batch_slice] = true_cfo
        toa_truth[batch_slice] = batch_toa_truth

        if collect_errors:
            wrapped_errors_samples = circular_delay_difference_samples(
                estimated_delay_samples=batch_toa_estimates[batch_detected],
                true_delay_samples=batch_toa_truth[batch_detected],
                span_samples=search_grid.delay_search_span_samples,
            )
            toa_errors_us.extend((wrapped_errors_samples / params.sample_rate_hz * 1e6).tolist())

        cursor += current_batch

    wrapped_toa_error_samples = circular_delay_difference_samples(
        estimated_delay_samples=toa_estimates,
        true_delay_samples=toa_truth,
        span_samples=search_grid.delay_search_span_samples,
    )

    summary = {
        "threshold": float(threshold),
        "misdetection_probability": float(np.mean(~detected)),
        "detection_probability": float(np.mean(detected)),
        "num_trials": int(num_trials),
        "num_detected": int(np.count_nonzero(detected)),
        "num_missed": int(np.count_nonzero(~detected)),
        "mean_statistic": float(np.mean(statistics)),
        "mean_abs_cfo_error_hz": float(np.mean(np.abs(cfo_estimates - cfo_truth))),
        "mean_abs_toa_error_us": float(
            np.mean(np.abs(wrapped_toa_error_samples) / params.sample_rate_hz * 1e6)
        ),
    }
    if collect_errors:
        summary["toa_errors_us"] = np.asarray(toa_errors_us, dtype=float)
    return summary
