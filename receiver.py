"""Receiver processing for NPRACH detection, ToA estimation, and CFO estimation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal

from config import DerivedParameters, SearchGrid, repetition_start_samples
from waveform import WaveformFrontEnd


@dataclass(frozen=True)
class SearchResult:
    """Single-trial output of a joint ToA/CFO search."""

    method: str
    statistic: float
    cfo_hz: float
    toa_samples: float
    p_index: int
    q_index: int
    surface: np.ndarray | None = None


def ensure_rx_dimension(symbols: np.ndarray) -> np.ndarray:
    """Ensure the symbol array has shape [rx, xi, L]."""
    symbols = np.asarray(symbols)
    if symbols.ndim == 2:
        return symbols[None, :, :]
    if symbols.ndim != 3:
        raise ValueError("Expected an array of shape [xi, L] or [rx, xi, L].")
    return symbols


def ensure_batch_rx_dimension(symbols: np.ndarray) -> np.ndarray:
    """Ensure the symbol array has shape [batch, rx, xi, L]."""
    symbols = np.asarray(symbols)
    if symbols.ndim == 3:
        return symbols[None, :, :, :]
    if symbols.ndim != 4:
        raise ValueError("Expected an array of shape [rx, xi, L] or [batch, rx, xi, L].")
    return symbols


def extract_hopped_symbols_from_waveform(
    params: DerivedParameters,
    rx_waveform: np.ndarray,
    hop_sequence: np.ndarray,
) -> np.ndarray:
    """
    Remove the CP, perform FFTs, and extract the hopped subcarrier.

    Output shape: [rx, xi, L]
    """
    rx_waveform = np.asarray(rx_waveform)
    if rx_waveform.ndim == 1:
        rx_waveform = rx_waveform[None, :]

    expected_samples = params.preamble_length * params.group_samples
    if rx_waveform.shape[1] < expected_samples:
        raise ValueError("The received waveform is shorter than the expected preamble capture.")
    if rx_waveform.shape[1] > expected_samples:
        rx_waveform = rx_waveform[:, :expected_samples]

    groups = rx_waveform.reshape(rx_waveform.shape[0], params.preamble_length, params.group_samples)
    useful = groups[:, :, params.cp_samples :]
    useful = useful.reshape(
        rx_waveform.shape[0],
        params.preamble_length,
        params.symbols_per_group,
        params.fft_size,
    )
    spectra = np.fft.fft(useful, axis=-1, norm="ortho")

    extracted = np.empty(
        (rx_waveform.shape[0], params.symbols_per_group, params.preamble_length),
        dtype=complex,
    )
    for group_index, active_bin in enumerate(np.asarray(hop_sequence, dtype=int)):
        extracted[:, :, group_index] = spectra[:, group_index, :, active_bin]
    return extracted


def extract_hopped_symbols_from_waveform_batch(
    params: DerivedParameters,
    rx_waveform: np.ndarray,
    hop_sequence: np.ndarray,
) -> np.ndarray:
    """
    Batch version of the CP removal, FFT, and hopped-subcarrier extraction.

    Input shape: [batch, rx, samples] or [rx, samples]
    Output shape: [batch, rx, xi, L]
    """
    rx_waveform = np.asarray(rx_waveform)
    if rx_waveform.ndim == 2:
        return extract_hopped_symbols_from_waveform(
            params=params,
            rx_waveform=rx_waveform,
            hop_sequence=hop_sequence,
        )[None, :, :, :]
    if rx_waveform.ndim != 3:
        raise ValueError("Expected [rx, samples] or [batch, rx, samples] waveform input.")

    batch_size = rx_waveform.shape[0]
    extracted = np.empty(
        (batch_size, rx_waveform.shape[1], params.symbols_per_group, params.preamble_length),
        dtype=complex,
    )
    for batch_index in range(batch_size):
        extracted[batch_index] = extract_hopped_symbols_from_waveform(
            params=params,
            rx_waveform=rx_waveform[batch_index],
            hop_sequence=hop_sequence,
        )
    return extracted


def receive_filter_and_downsample(
    rx_waveform_oversampled: np.ndarray,
    front_end: WaveformFrontEnd,
) -> np.ndarray:
    """
    Apply the explicit RX anti-alias filter and downsample back to the base rate.
    """
    rx_waveform_oversampled = np.asarray(rx_waveform_oversampled)
    if rx_waveform_oversampled.ndim == 1:
        rx_waveform_oversampled = rx_waveform_oversampled[None, :]

    return signal.resample_poly(
        rx_waveform_oversampled,
        up=1,
        down=front_end.oversampling_factor,
        window=front_end.coefficients,
        padtype="constant",
        axis=-1,
    )


def process_received_waveform_to_symbols(
    params: DerivedParameters,
    rx_waveform_oversampled: np.ndarray,
    front_end: WaveformFrontEnd,
    hop_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Complete the waveform-domain RX front-end up to the extracted symbols.

    Returns:
    - downsampled base-rate waveform capture
    - extracted hopped-subcarrier observations of shape [rx, xi, L]
    """
    rx_waveform_base = receive_filter_and_downsample(
        rx_waveform_oversampled=rx_waveform_oversampled,
        front_end=front_end,
    )
    extracted = extract_hopped_symbols_from_waveform(
        params=params,
        rx_waveform=rx_waveform_base,
        hop_sequence=hop_sequence,
    )
    return rx_waveform_base, extracted


def process_received_waveform_batch_to_symbols(
    params: DerivedParameters,
    rx_waveform_oversampled: np.ndarray,
    front_end: WaveformFrontEnd,
    hop_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch version of the waveform-domain RX front-end.

    Input shape: [batch, rx, oversampled_samples]
    Returns:
    - downsampled base-rate captures [batch, rx, base_samples]
    - extracted observations [batch, rx, xi, L]
    """
    rx_waveform_base = receive_filter_and_downsample(
        rx_waveform_oversampled=rx_waveform_oversampled,
        front_end=front_end,
    )
    extracted = extract_hopped_symbols_from_waveform_batch(
        params=params,
        rx_waveform=rx_waveform_base,
        hop_sequence=hop_sequence,
    )
    return rx_waveform_base, extracted


def form_z(received_symbols: np.ndarray) -> np.ndarray:
    """Form z[i, m] = y_tilde[i, m] * u*[m]; here u[m] = 1 by design."""
    return np.asarray(received_symbols, dtype=complex).copy()


def direct_joint_search(
    params: DerivedParameters,
    hop_sequence: np.ndarray,
    z_symbols: np.ndarray,
    search_grid: SearchGrid,
    return_surface: bool = False,
) -> SearchResult:
    """
    Evaluate Eq. (4)-(5) by direct brute-force search over D and Delta_f.
    """
    hop_sequence = np.asarray(hop_sequence, dtype=float)
    z = ensure_rx_dimension(z_symbols)
    time_samples = repetition_start_samples(params)

    cfo_basis = np.exp(
        -1j
        * 2.0
        * np.pi
        * search_grid.cfo_candidates_normalized[:, None, None]
        * time_samples[None, :, :]
    )
    toa_basis = np.exp(
        1j
        * 2.0
        * np.pi
        * search_grid.toa_candidates_samples[:, None]
        * hop_sequence[None, :]
        / params.fft_size
    )

    objective = np.zeros((search_grid.M1, search_grid.M2), dtype=float)
    for block_index in range(params.preamble_length // params.groups_per_block):
        start = block_index * params.groups_per_block
        stop = start + params.groups_per_block
        block = z[:, :, start:stop]
        cfo_block = cfo_basis[:, :, start:stop]
        toa_block = toa_basis[:, start:stop]

        collapsed = np.einsum("rim,pim->rpm", block, cfo_block, optimize=True)
        jg = np.einsum("rpm,dm->rpd", collapsed, toa_block, optimize=True)
        objective += np.sum(np.abs(jg) ** 2, axis=0)

    p_index, q_index = np.unravel_index(np.argmax(objective), objective.shape)
    return SearchResult(
        method="direct",
        statistic=float(objective[p_index, q_index]),
        cfo_hz=float(search_grid.cfo_candidates_hz[p_index]),
        toa_samples=float(search_grid.toa_candidates_samples[q_index]),
        p_index=int(p_index),
        q_index=int(q_index),
        surface=objective if return_surface else None,
    )


def fft_joint_search_batch(
    params: DerivedParameters,
    hop_sequence: np.ndarray,
    z_symbols: np.ndarray,
    search_grid: SearchGrid,
    return_surface: bool = False,
) -> dict:
    """
    Evaluate Eq. (6) in batch form.

    Input shape: [batch, rx, xi, L]
    """
    hop_sequence = np.asarray(hop_sequence, dtype=int)
    z = ensure_batch_rx_dimension(z_symbols)
    batch_size, num_rx, _, _ = z.shape

    time_basis, frequency_basis = _eq6_factorized_bases(
        M1=search_grid.M1,
        M2=search_grid.M2,
        symbols_per_group=params.symbols_per_group,
        groups_per_block=params.groups_per_block,
        band_subcarriers=params.band_subcarriers,
    )

    objective = np.zeros((batch_size, search_grid.M1, search_grid.M2), dtype=float)
    for block_index in range(params.preamble_length // params.groups_per_block):
        start = block_index * params.groups_per_block
        stop = start + params.groups_per_block
        z_block = z[:, :, :, start:stop]
        time_projected = np.einsum("brig,pig->brpg", z_block, time_basis, optimize=True)
        block_frequency_basis = frequency_basis[:, hop_sequence[start:stop]]
        W = np.einsum("brpg,qg->brpq", time_projected, block_frequency_basis, optimize=True)
        objective += np.sum(np.abs(W) ** 2, axis=1)

    flattened = objective.reshape(batch_size, -1)
    flat_indices = np.argmax(flattened, axis=1)
    p_indices = flat_indices // search_grid.M2
    q_indices = flat_indices % search_grid.M2
    statistics = flattened[np.arange(batch_size), flat_indices]

    result = {
        "statistics": statistics.astype(float),
        "p_indices": p_indices.astype(int),
        "q_indices": q_indices.astype(int),
        "cfo_hz": search_grid.cfo_candidates_hz[p_indices].astype(float),
        "toa_samples": search_grid.toa_candidates_samples[q_indices].astype(float),
    }
    if return_surface:
        result["surface"] = objective
    return result


def fft_joint_search(
    params: DerivedParameters,
    hop_sequence: np.ndarray,
    z_symbols: np.ndarray,
    search_grid: SearchGrid,
    return_surface: bool = False,
) -> SearchResult:
    """Single-trial wrapper around the batch 2-D FFT estimator."""
    output = fft_joint_search_batch(
        params=params,
        hop_sequence=hop_sequence,
        z_symbols=ensure_rx_dimension(z_symbols)[None, :, :, :],
        search_grid=search_grid,
        return_surface=return_surface,
    )
    surface = output["surface"][0] if return_surface else None
    return SearchResult(
        method="fft",
        statistic=float(output["statistics"][0]),
        cfo_hz=float(output["cfo_hz"][0]),
        toa_samples=float(output["toa_samples"][0]),
        p_index=int(output["p_indices"][0]),
        q_index=int(output["q_indices"][0]),
        surface=surface,
    )


def compare_search_results(direct: SearchResult, fft_result: SearchResult) -> dict:
    """Compare the direct and FFT search results on the same observation."""
    comparison = {
        "same_p_index": direct.p_index == fft_result.p_index,
        "same_q_index": direct.q_index == fft_result.q_index,
        "cfo_hz_difference": float(abs(direct.cfo_hz - fft_result.cfo_hz)),
        "toa_samples_difference": float(abs(direct.toa_samples - fft_result.toa_samples)),
        "statistic_difference": float(abs(direct.statistic - fft_result.statistic)),
    }
    if direct.surface is not None and fft_result.surface is not None:
        direct_normalized = direct.surface / np.max(direct.surface)
        fft_normalized = fft_result.surface / np.max(fft_result.surface)
        comparison["normalized_surface_max_error"] = float(
            np.max(np.abs(direct_normalized - fft_normalized))
        )
    return comparison


@lru_cache(maxsize=None)
def _eq6_factorized_bases(
    M1: int,
    M2: int,
    symbols_per_group: int,
    groups_per_block: int,
    band_subcarriers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute the separable Eq. (6) phase bases.

    This is mathematically identical to forming the sparse wg[n, k] array and
    applying a 2-D FFT, but avoids repeatedly creating a mostly zero-valued
    M1 x M2 grid for every block and Monte Carlo batch.
    """
    local_groups = np.arange(groups_per_block)[None, :]
    repetitions = np.arange(symbols_per_group)[:, None]
    time_indices = local_groups * (symbols_per_group + 1) + repetitions

    p = np.arange(M1)[:, None, None]
    time_basis = np.exp(-1j * 2.0 * np.pi * p * time_indices[None, :, :] / M1)

    q = np.arange(M2)[:, None]
    k = np.arange(band_subcarriers)[None, :]
    frequency_basis = np.exp(-1j * 2.0 * np.pi * q * k / M2)
    return time_basis, frequency_basis
