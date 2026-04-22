"""Waveform-domain channel and noise models for NPRACH simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    DerivedParameters,
    TYPICAL_URBAN_APPROX_DELAYS_US,
    TYPICAL_URBAN_APPROX_POWERS_DB,
)
from waveform import WaveformFrontEnd


@dataclass(frozen=True)
class WaveformChannelRealization:
    """Waveform-domain receive realization."""

    rx_waveform_oversampled: np.ndarray
    true_delay_samples: float
    true_delay_upsampled_samples: int
    tap_delays_upsampled_samples: np.ndarray
    tap_powers_linear: np.ndarray
    noise_variance_input: float


def complex_gaussian(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    variance: float = 1.0,
) -> np.ndarray:
    """Generate circularly symmetric complex Gaussian noise."""
    sigma = np.sqrt(variance / 2.0)
    return sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def noise_variance_from_snr_db(snr_db: float) -> float:
    """Convert SNR in dB into the extracted-symbol noise variance target."""
    return 10.0 ** (-snr_db / 10.0)


def typical_urban_profile(
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the explicit urban-like tapped-delay profile used by this project.

    This is a documented engineering approximation, not a claim of the exact
    hidden 3GPP channel profile used in the paper's simulator.
    """
    delays_seconds = 1e-6 * np.asarray(TYPICAL_URBAN_APPROX_DELAYS_US, dtype=float)
    delays_samples = delays_seconds * sample_rate_hz
    powers_linear = 10.0 ** (np.asarray(TYPICAL_URBAN_APPROX_POWERS_DB, dtype=float) / 10.0)
    powers_linear /= np.sum(powers_linear)
    return delays_samples, powers_linear


def _waveform_tap_model(
    sample_rate_hz: float,
    channel_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tap delays and powers for the waveform-domain channel."""
    if channel_model == "unit_gain":
        return np.array([0], dtype=int), np.array([1.0], dtype=float)
    if channel_model == "flat_fading":
        return np.array([0], dtype=int), np.array([1.0], dtype=float)
    if channel_model == "typical_urban":
        delays_samples, powers_linear = typical_urban_profile(sample_rate_hz)
        return np.rint(delays_samples).astype(int), powers_linear.astype(float)
    raise ValueError(
        "Waveform-domain channel supports 'unit_gain', 'flat_fading', and 'typical_urban'."
    )


def apply_channel_to_waveform(
    params: DerivedParameters,
    tx_waveform_upsampled: np.ndarray,
    front_end: WaveformFrontEnd,
    toa_samples: float,
    cfo_hz: float,
    drift_hz_per_s: float,
    input_noise_variance: float | None,
    rng: np.random.Generator,
    num_rx: int = 1,
    channel_model: str = "typical_urban",
    doppler_hz: float = 0.0,
) -> WaveformChannelRealization:
    """
    Apply the waveform-domain channel to an oversampled NPRACH waveform.

    The timing offset is implemented on the oversampled grid so that the
    reference path includes a fractional-delay approximation at 1 / U of the
    base sample period.
    """
    sample_rate_hz = front_end.upsampled_sample_rate_hz
    true_delay_upsampled_samples = int(round(toa_samples * front_end.oversampling_factor))
    true_delay_samples = true_delay_upsampled_samples / front_end.oversampling_factor

    tap_delays_upsampled_samples, tap_powers_linear = _waveform_tap_model(
        sample_rate_hz=sample_rate_hz,
        channel_model=channel_model,
    )
    num_taps = tap_delays_upsampled_samples.size
    max_tap_delay = int(np.max(tap_delays_upsampled_samples))

    output_length = tx_waveform_upsampled.size + true_delay_upsampled_samples + max_tap_delay
    time_axis = np.arange(output_length) / sample_rate_hz

    static_coefficients = complex_gaussian((num_rx, num_taps), rng=rng, variance=1.0)
    static_coefficients *= np.sqrt(tap_powers_linear)[None, :]
    if channel_model == "unit_gain":
        static_coefficients[:] = 1.0 + 0.0j

    if doppler_hz != 0.0:
        doppler_shifts = rng.uniform(-doppler_hz, doppler_hz, size=(num_rx, num_taps))
    else:
        doppler_shifts = np.zeros((num_rx, num_taps), dtype=float)

    rx_waveform = np.zeros((num_rx, output_length), dtype=complex)
    for tap_index, tap_delay in enumerate(tap_delays_upsampled_samples):
        start = true_delay_upsampled_samples + int(tap_delay)
        stop = start + tx_waveform_upsampled.size
        tap_phase = np.exp(
            1j * 2.0 * np.pi * doppler_shifts[:, tap_index, None] * time_axis[None, start:stop]
        )
        rx_waveform[:, start:stop] += (
            static_coefficients[:, tap_index, None] * tap_phase * tx_waveform_upsampled[None, :]
        )

    cfo_phase = np.exp(
        1j
        * 2.0
        * np.pi
        * (cfo_hz * time_axis + 0.5 * drift_hz_per_s * time_axis**2)
    )
    rx_waveform *= cfo_phase[None, :]

    noise_variance = 0.0 if input_noise_variance is None else float(input_noise_variance)
    if noise_variance > 0.0:
        rx_waveform += complex_gaussian(rx_waveform.shape, rng=rng, variance=noise_variance)

    return WaveformChannelRealization(
        rx_waveform_oversampled=rx_waveform,
        true_delay_samples=float(true_delay_samples),
        true_delay_upsampled_samples=int(true_delay_upsampled_samples),
        tap_delays_upsampled_samples=tap_delays_upsampled_samples,
        tap_powers_linear=tap_powers_linear,
        noise_variance_input=noise_variance,
    )


def generate_waveform_noise_only_input(
    params: DerivedParameters,
    front_end: WaveformFrontEnd,
    input_noise_variance: float,
    rng: np.random.Generator,
    num_rx: int = 1,
    batch_size: int = 1,
    extra_tail_samples: int = 0,
) -> np.ndarray:
    """Generate Gaussian noise at the receiver input for false alarm testing."""
    total_base_samples = params.preamble_length * params.group_samples + extra_tail_samples
    total_upsampled_samples = total_base_samples * front_end.oversampling_factor
    noise = complex_gaussian(
        (batch_size, num_rx, total_upsampled_samples),
        rng=rng,
        variance=input_noise_variance,
    )
    if batch_size == 1:
        return noise[0]
    return noise
