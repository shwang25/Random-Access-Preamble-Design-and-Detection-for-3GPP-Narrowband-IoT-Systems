"""Frequency hopping logic for the NPRACH preamble."""

from __future__ import annotations

import numpy as np

from config import DEFAULT_SEED


INNER_HOP_STEPS = (1, 6, 1)
INNER_HOP_OFFSETS = np.array((0, 1, 7, 8), dtype=int)


def pseudo_random_outer_offsets(
    num_blocks: int,
    band_subcarriers: int,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate deterministic pseudo-random block offsets.

    Approximation note:
    The paper explains that the outer-layer hopping is pseudo-random but does
    not publish the exact standardized sequence in the letter. This function
    uses a fixed-seed permutation stream so the simulator remains reproducible.
    """
    rng = np.random.default_rng(seed)
    offsets = [0]
    while len(offsets) < num_blocks:
        offsets.extend(int(value) for value in rng.permutation(band_subcarriers))
    return np.asarray(offsets[:num_blocks], dtype=int)


def generate_hopping_sequence(
    preamble_length: int,
    band_subcarriers: int,
    preamble_index: int = 0,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Generate the full per-group subcarrier sequence Omega(m).

    Inner-layer pattern within each block of four groups:
        0, +1, +6, +1  -> relative offsets 0, 1, 7, 8
    """
    if preamble_length % 4 != 0:
        raise ValueError("The preamble length must be a multiple of 4 symbol groups.")

    num_blocks = preamble_length // 4
    outer_offsets = pseudo_random_outer_offsets(num_blocks, band_subcarriers, seed=seed)

    hops = np.empty(preamble_length, dtype=int)
    for block_index in range(num_blocks):
        base = (preamble_index + outer_offsets[block_index]) % band_subcarriers
        block_bins = (base + INNER_HOP_OFFSETS) % band_subcarriers
        start = 4 * block_index
        hops[start : start + 4] = block_bins
    return hops


def modular_differences(sequence: np.ndarray, modulus: int) -> np.ndarray:
    """Return modular forward differences."""
    sequence = np.asarray(sequence, dtype=int)
    return np.mod(np.diff(sequence), modulus)


def validate_hopping_sequence(sequence: np.ndarray, band_subcarriers: int) -> dict:
    """
    Validate the generated hopping sequence.

    Checks:
    - all subcarrier indices are valid
    - every block of four groups follows the modular 1-6-1 rule
    """
    sequence = np.asarray(sequence, dtype=int)
    if sequence.size % 4 != 0:
        raise ValueError("The hopping sequence must contain a multiple of 4 entries.")

    num_blocks = sequence.size // 4
    per_block_differences = []
    pattern_ok = True

    for block_index in range(num_blocks):
        block = sequence[4 * block_index : 4 * (block_index + 1)]
        diffs = modular_differences(block, band_subcarriers).tolist()
        per_block_differences.append(diffs)
        if tuple(diffs) != INNER_HOP_STEPS:
            pattern_ok = False

    return {
        "valid_indices": bool(np.all((0 <= sequence) & (sequence < band_subcarriers))),
        "pattern_ok": pattern_ok,
        "per_block_differences": per_block_differences,
        "sequence": sequence.tolist(),
    }


def example_sequence_string(sequence: np.ndarray, max_length: int = 16) -> str:
    """Format a concise example hopping sequence for logging."""
    sequence = np.asarray(sequence, dtype=int)
    clipped = sequence[:max_length].tolist()
    suffix = "" if sequence.size <= max_length else " ..."
    return f"{clipped}{suffix}"
