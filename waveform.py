"""Waveform generation and front-end filtering for the single-tone NPRACH preamble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from config import (
    DEFAULT_SEED,
    DerivedParameters,
    WAVEFORM_FILTER_CUTOFF_FACTOR,
    WAVEFORM_FILTER_TAPS,
    WAVEFORM_FILTER_WINDOW,
    WAVEFORM_OVERSAMPLING_FACTOR,
)
from hopping import generate_hopping_sequence


@dataclass(frozen=True)
class WaveformFrontEnd:
    """Explicit interpolation / decimation front-end configuration."""

    oversampling_factor: int
    upsampled_sample_rate_hz: float
    num_taps: int
    cutoff_hz: float
    coefficients: np.ndarray
    group_delay_upsampled_samples: int


@dataclass(frozen=True)
class WaveformMetadata:
    """Metadata associated with a generated NPRACH waveform."""

    hop_sequence: np.ndarray
    group_start_samples: np.ndarray
    useful_start_samples: np.ndarray
    waveform_length_samples: int
    oversampling_factor: int
    tx_waveform_length_samples: int


@dataclass(frozen=True)
class PreambleWaveformReference:
    """Cached reference waveform for one NPRACH preamble signature."""

    preamble_index: int
    hop_sequence: np.ndarray
    resource_grid: np.ndarray
    baseband_waveform: np.ndarray
    tx_waveform_upsampled: np.ndarray
    metadata: WaveformMetadata


def active_subcarrier_spectrum(
    fft_size: int,
    active_bin: int,
    symbol_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Create a frequency-domain vector with a single active subcarrier."""
    spectrum = np.zeros(fft_size, dtype=complex)
    spectrum[int(active_bin) % fft_size] = symbol_value
    return spectrum


def design_front_end_filter(
    params: DerivedParameters,
    oversampling_factor: int = WAVEFORM_OVERSAMPLING_FACTOR,
    num_taps: int = WAVEFORM_FILTER_TAPS,
    cutoff_factor: float = WAVEFORM_FILTER_CUTOFF_FACTOR,
) -> WaveformFrontEnd:
    """
    Design the explicit FIR front-end filter used for TX interpolation and RX decimation.

    Engineering approximation:
    The paper states "upsampling and filtering" but does not specify the exact
    interpolation filters. This implementation uses a linear-phase Kaiser-window
    low-pass FIR. The same FIR is reused at the receiver as an anti-alias filter.
    SciPy's `resample_poly` is then used for interpolation / decimation with the
    designed FIR, which compensates the fixed filter group delay automatically.
    """
    if num_taps % 2 == 0:
        raise ValueError("The front-end FIR must use an odd number of taps.")

    upsampled_sample_rate_hz = params.sample_rate_hz * oversampling_factor
    cutoff_hz = cutoff_factor * params.sample_rate_hz
    coefficients = signal.firwin(
        numtaps=num_taps,
        cutoff=cutoff_hz,
        window=WAVEFORM_FILTER_WINDOW,
        fs=upsampled_sample_rate_hz,
    )
    return WaveformFrontEnd(
        oversampling_factor=oversampling_factor,
        upsampled_sample_rate_hz=upsampled_sample_rate_hz,
        num_taps=num_taps,
        cutoff_hz=float(cutoff_hz),
        coefficients=coefficients.astype(float),
        group_delay_upsampled_samples=(num_taps - 1) // 2,
    )


def build_nprach_resource_grid(
    params: DerivedParameters,
    hop_sequence: np.ndarray,
    symbol_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """
    Build the OFDM resource grid for the full preamble.

    Output shape: [L, xi, N]
    """
    hop_sequence = np.asarray(hop_sequence, dtype=int)
    if hop_sequence.size != params.preamble_length:
        raise ValueError("The hopping sequence length must match the preamble length L.")

    grid = np.zeros(
        (params.preamble_length, params.symbols_per_group, params.fft_size),
        dtype=complex,
    )
    for group_index, active_bin in enumerate(hop_sequence):
        spectrum = active_subcarrier_spectrum(
            fft_size=params.fft_size,
            active_bin=int(active_bin),
            symbol_value=symbol_value,
        )
        grid[group_index, :, :] = spectrum[None, :]
    return grid


def generate_useful_symbol(
    params: DerivedParameters,
    active_bin: int,
    symbol_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """
    Generate one useful N-sample OFDM symbol for the given active subcarrier.
    """
    spectrum = active_subcarrier_spectrum(params.fft_size, active_bin, symbol_value=symbol_value)
    return np.fft.ifft(spectrum, n=params.fft_size, norm="ortho")


def build_symbol_group(
    params: DerivedParameters,
    active_bin: int,
    symbol_value: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """
    Build one NPRACH symbol group: one CP followed by five repeated symbols.
    """
    useful_symbol = generate_useful_symbol(params, active_bin, symbol_value=symbol_value)
    useful_block = np.tile(useful_symbol, params.symbols_per_group)
    cyclic_prefix = useful_block[-params.cp_samples :]
    return np.concatenate((cyclic_prefix, useful_block))


def generate_preamble_waveform_from_grid(
    params: DerivedParameters,
    resource_grid: np.ndarray,
) -> np.ndarray:
    """
    Generate the base-rate time-domain NPRACH waveform from a resource grid.
    """
    useful_symbols = np.fft.ifft(resource_grid, axis=-1, norm="ortho")
    groups = []
    for group_index in range(params.preamble_length):
        useful_block = useful_symbols[group_index].reshape(-1)
        cyclic_prefix = useful_block[-params.cp_samples :]
        groups.append(np.concatenate((cyclic_prefix, useful_block)))
    return np.concatenate(groups)


def upsample_and_filter_waveform(
    waveform: np.ndarray,
    front_end: WaveformFrontEnd,
) -> np.ndarray:
    """
    Apply the explicit TX interpolation stage.
    """
    return signal.resample_poly(
        waveform,
        up=front_end.oversampling_factor,
        down=1,
        window=front_end.coefficients,
        padtype="constant",
    )


def generate_preamble_waveform(
    params: DerivedParameters,
    hop_sequence: np.ndarray,
    symbol_value: complex = 1.0 + 0.0j,
) -> tuple[np.ndarray, WaveformMetadata]:
    """
    Generate the full base-rate time-domain NPRACH waveform for one preamble.
    """
    resource_grid = build_nprach_resource_grid(params, hop_sequence, symbol_value=symbol_value)
    waveform = generate_preamble_waveform_from_grid(params, resource_grid)
    group_starts = np.arange(params.preamble_length) * params.group_samples
    useful_starts = group_starts + params.cp_samples

    metadata = WaveformMetadata(
        hop_sequence=np.asarray(hop_sequence, dtype=int).copy(),
        group_start_samples=group_starts,
        useful_start_samples=useful_starts,
        waveform_length_samples=waveform.size,
        oversampling_factor=1,
        tx_waveform_length_samples=waveform.size,
    )
    return waveform, metadata


def generate_preamble_reference(
    params: DerivedParameters,
    preamble_index: int,
    front_end: WaveformFrontEnd,
    hopping_seed: int = DEFAULT_SEED,
    symbol_value: complex = 1.0 + 0.0j,
) -> PreambleWaveformReference:
    """
    Generate and cache the baseband and oversampled TX waveform for one preamble.
    """
    hop_sequence = generate_hopping_sequence(
        preamble_length=params.preamble_length,
        band_subcarriers=params.band_subcarriers,
        preamble_index=preamble_index,
        seed=hopping_seed,
    )
    resource_grid = build_nprach_resource_grid(params, hop_sequence, symbol_value=symbol_value)
    baseband_waveform = generate_preamble_waveform_from_grid(params, resource_grid)
    tx_waveform_upsampled = upsample_and_filter_waveform(baseband_waveform, front_end=front_end)
    group_starts = np.arange(params.preamble_length) * params.group_samples
    useful_starts = group_starts + params.cp_samples

    metadata = WaveformMetadata(
        hop_sequence=hop_sequence.copy(),
        group_start_samples=group_starts,
        useful_start_samples=useful_starts,
        waveform_length_samples=baseband_waveform.size,
        oversampling_factor=front_end.oversampling_factor,
        tx_waveform_length_samples=tx_waveform_upsampled.size,
    )
    return PreambleWaveformReference(
        preamble_index=preamble_index,
        hop_sequence=hop_sequence,
        resource_grid=resource_grid,
        baseband_waveform=baseband_waveform,
        tx_waveform_upsampled=tx_waveform_upsampled,
        metadata=metadata,
    )


def build_preamble_reference_bank(
    params: DerivedParameters,
    front_end: WaveformFrontEnd,
    hopping_seed: int = DEFAULT_SEED,
) -> dict[int, PreambleWaveformReference]:
    """
    Build a reference bank for all configured orthogonal preamble signatures.
    """
    return {
        preamble_index: generate_preamble_reference(
            params=params,
            preamble_index=preamble_index,
            front_end=front_end,
            hopping_seed=hopping_seed,
        )
        for preamble_index in range(params.band_subcarriers)
    }


def waveform_summary(
    params: DerivedParameters,
    waveform: np.ndarray,
    hop_sequence: np.ndarray,
    front_end: WaveformFrontEnd | None = None,
) -> dict:
    """Return a concise waveform summary for validation logs."""
    summary = {
        "waveform_samples": int(waveform.size),
        "waveform_duration_ms": 1e3 * waveform.size / params.sample_rate_hz,
        "group_samples": params.group_samples,
        "preamble_length": params.preamble_length,
        "subcarrier_evolution": np.asarray(hop_sequence, dtype=int).tolist(),
    }
    if front_end is not None:
        summary.update(
            {
                "oversampling_factor": front_end.oversampling_factor,
                "upsampled_sample_rate_hz": front_end.upsampled_sample_rate_hz,
                "front_end_filter_taps": front_end.num_taps,
                "front_end_filter_cutoff_hz": front_end.cutoff_hz,
            }
        )
    return summary
