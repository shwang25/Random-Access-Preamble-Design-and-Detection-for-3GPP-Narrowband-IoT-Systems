"""Configuration and parameter derivation for the NPRACH simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULT_COMPATIBILITY_KEYS = ("mode", "channel_model", "m1", "m2", "num_rx")

DEFAULT_SEED = 20260422

BANDWIDTH_HZ = 180_000.0
SUBCARRIER_SPACING_HZ = 3_750.0
CP_DURATION_S = 266.7e-6
SYMBOLS_PER_GROUP = 5
GROUPS_PER_BLOCK = 4
NPRACH_BAND_SUBCARRIERS = 12

DEFAULT_PREAMBLE_LENGTHS = (8, 32, 128)
PAPER_SNR_DB = {8: 14.25, 32: 4.25, 128: -5.75}
PAPER_CFO_RANGE_HZ = (-50.0, 50.0)
PAPER_DRIFT_RANGE_HZ_PER_S = (-22.5, 22.5)
PAPER_DOPPLER_HZ = 1.0
PAPER_NUM_RX = 2
PAPER_ITERATIONS_DETECTION = 10_000
PAPER_ITERATIONS_FALSE_ALARM = 100_000
TARGET_PFA = 1e-3

DEFAULT_SEARCH_M1 = 128
DEFAULT_SEARCH_M2 = 256
DEFAULT_SIMULATION_MODE = "waveform"
DEFAULT_PAPER_CHANNEL_MODEL = "typical_urban"
DEFAULT_FAST_CHANNEL_MODEL = "block_fading"

WAVEFORM_OVERSAMPLING_FACTOR = 8
WAVEFORM_FILTER_TAPS = 65
WAVEFORM_FILTER_CUTOFF_FACTOR = 0.45
WAVEFORM_FILTER_WINDOW = ("kaiser", 8.0)

# This is a compact, explicitly documented urban-like multipath approximation.
# It is not claimed to be the exact standardized 3GPP Typical Urban profile.
TYPICAL_URBAN_APPROX_DELAYS_US = (0.0, 0.5, 1.5, 2.5, 3.5, 5.0)
TYPICAL_URBAN_APPROX_POWERS_DB = (0.0, -1.5, -3.0, -6.0, -9.0, -12.0)


@dataclass(frozen=True)
class CoverageCase:
    """Paper coverage-class abstraction."""

    preamble_length: int
    snr_db: float


@dataclass(frozen=True)
class DerivedParameters:
    """Derived NPRACH parameters used throughout the simulator."""

    bandwidth_hz: float
    subcarrier_spacing_hz: float
    cp_duration_s_requested: float
    symbols_per_group: int
    groups_per_block: int
    preamble_length: int
    band_subcarriers: int
    fft_size: int
    sample_rate_hz: float
    cp_samples: int
    cp_duration_s_derived: float
    useful_symbol_duration_s: float
    group_samples: int
    group_duration_s: float
    basic_unit_duration_s: float
    preamble_duration_s: float

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class SearchGrid:
    """Search grid shared by the direct and 2-D FFT estimators."""

    M1: int
    M2: int
    cfo_candidates_hz: np.ndarray
    cfo_candidates_normalized: np.ndarray
    toa_candidates_principal_samples: np.ndarray
    toa_candidates_samples: np.ndarray
    toa_candidates_seconds: np.ndarray
    delay_search_span_samples: float

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation."""
        return {
            "M1": self.M1,
            "M2": self.M2,
            "cfo_candidates_hz": self.cfo_candidates_hz.tolist(),
            "cfo_candidates_normalized": self.cfo_candidates_normalized.tolist(),
            "toa_candidates_principal_samples": self.toa_candidates_principal_samples.tolist(),
            "toa_candidates_samples": self.toa_candidates_samples.tolist(),
            "toa_candidates_seconds": self.toa_candidates_seconds.tolist(),
            "delay_search_span_samples": self.delay_search_span_samples,
        }


def ensure_results_dir() -> Path:
    """Create the results directory if it does not exist."""
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


def ensure_mode_results_dir(mode: str) -> Path:
    """Create and return the mode-specific results directory."""
    path = ensure_results_dir() / mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_path(mode: str, filename: str) -> Path:
    """Return the default output path for one simulation mode."""
    return ensure_mode_results_dir(mode) / filename


def build_run_configuration(
    mode: str,
    channel_model: str,
    m1: int,
    m2: int,
    num_rx: int,
) -> dict:
    """Build the shared run-configuration metadata stored with result files."""
    return {
        "mode": mode,
        "channel_model": channel_model,
        "m1": int(m1),
        "m2": int(m2),
        "num_rx": int(num_rx),
    }


def run_configuration_matches(
    metadata: dict,
    expected_run_configuration: dict,
    required_keys: tuple[str, ...] = RESULT_COMPATIBILITY_KEYS,
) -> bool:
    """Return whether a metadata blob matches the expected run configuration."""
    run_configuration = metadata.get("run_config")
    if not isinstance(run_configuration, dict):
        return False
    return all(run_configuration.get(key) == expected_run_configuration.get(key) for key in required_keys)


def incompatible_run_configuration_keys(
    reference_run_configuration: dict,
    candidate_run_configuration: dict,
    required_keys: tuple[str, ...] = RESULT_COMPATIBILITY_KEYS,
) -> list[str]:
    """Return the required run-configuration keys that do not match."""
    return [
        key
        for key in required_keys
        if reference_run_configuration.get(key) != candidate_run_configuration.get(key)
    ]


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Create a reproducible NumPy random number generator."""
    return np.random.default_rng(seed)


def paper_coverage_cases() -> list[CoverageCase]:
    """Return the paper's three NPRACH coverage classes."""
    return [CoverageCase(length, PAPER_SNR_DB[length]) for length in DEFAULT_PREAMBLE_LENGTHS]


def smallest_power_of_two_at_least(value: float) -> int:
    """Return the smallest power of two greater than or equal to value."""
    if value <= 1:
        return 1
    return 1 << math.ceil(math.log2(value))


def derive_parameters(
    preamble_length: int,
    bandwidth_hz: float = BANDWIDTH_HZ,
    subcarrier_spacing_hz: float = SUBCARRIER_SPACING_HZ,
    cp_duration_s: float = CP_DURATION_S,
    symbols_per_group: int = SYMBOLS_PER_GROUP,
    groups_per_block: int = GROUPS_PER_BLOCK,
    band_subcarriers: int = NPRACH_BAND_SUBCARRIERS,
    fft_size: int | None = None,
) -> DerivedParameters:
    """
    Derive discrete-time parameters from the paper settings.

    The paper does not state an explicit FFT size, so we choose the smallest
    power-of-two FFT whose occupied raster can host the 180 kHz resource grid.
    """
    if preamble_length % groups_per_block != 0:
        raise ValueError("The preamble length must be a multiple of the block size Q = 4.")

    if fft_size is None:
        minimum_fft = bandwidth_hz / subcarrier_spacing_hz
        fft_size = smallest_power_of_two_at_least(minimum_fft)

    sample_rate_hz = fft_size * subcarrier_spacing_hz
    cp_samples = int(round(cp_duration_s * sample_rate_hz))
    cp_duration_s_derived = cp_samples / sample_rate_hz
    useful_symbol_duration_s = fft_size / sample_rate_hz
    group_samples = cp_samples + symbols_per_group * fft_size
    group_duration_s = group_samples / sample_rate_hz
    basic_unit_duration_s = groups_per_block * group_duration_s
    preamble_duration_s = preamble_length * group_duration_s

    return DerivedParameters(
        bandwidth_hz=bandwidth_hz,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        cp_duration_s_requested=cp_duration_s,
        symbols_per_group=symbols_per_group,
        groups_per_block=groups_per_block,
        preamble_length=preamble_length,
        band_subcarriers=band_subcarriers,
        fft_size=fft_size,
        sample_rate_hz=sample_rate_hz,
        cp_samples=cp_samples,
        cp_duration_s_derived=cp_duration_s_derived,
        useful_symbol_duration_s=useful_symbol_duration_s,
        group_samples=group_samples,
        group_duration_s=group_duration_s,
        basic_unit_duration_s=basic_unit_duration_s,
        preamble_duration_s=preamble_duration_s,
    )


def build_search_grid(
    params: DerivedParameters,
    M1: int = DEFAULT_SEARCH_M1,
    M2: int = DEFAULT_SEARCH_M2,
) -> SearchGrid:
    """
    Build the shared search grid used by the direct and 2-D FFT estimators.

    The CFO grid follows the paper's Eq. (6) mapping:
        Delta_f = wrapped_p / (N * M1)

    The ToA grid follows the paper's Eq. (6) principal-value mapping first,
    then wraps it into the physical delay domain [0, Ncp). This explicit split
    prevents sign-convention bugs when converting between estimation and error
    reporting domains.
    """
    if M1 % 2 != 0 or M2 % 2 != 0:
        raise ValueError("M1 and M2 must be even so that the principal-value mapping is well defined.")

    p = np.arange(M1)
    wrapped_p = np.where(p < M1 // 2, p, p - M1)
    cfo_candidates_normalized = wrapped_p / (params.fft_size * M1)
    cfo_candidates_hz = cfo_candidates_normalized * params.sample_rate_hz

    q = np.arange(M2)
    toa_candidates_principal_samples = toa_principal_from_q_indices(
        q_indices=q,
        params=params,
        M2=M2,
    )
    toa_candidates_samples = wrap_delay_samples(
        delay_samples=toa_candidates_principal_samples,
        span_samples=params.cp_samples,
    )
    toa_candidates_seconds = toa_candidates_samples / params.sample_rate_hz

    return SearchGrid(
        M1=M1,
        M2=M2,
        cfo_candidates_hz=cfo_candidates_hz.astype(float),
        cfo_candidates_normalized=cfo_candidates_normalized.astype(float),
        toa_candidates_principal_samples=toa_candidates_principal_samples.astype(float),
        toa_candidates_samples=toa_candidates_samples.astype(float),
        toa_candidates_seconds=toa_candidates_seconds.astype(float),
        delay_search_span_samples=float(params.cp_samples),
    )


def repetition_start_samples(params: DerivedParameters) -> np.ndarray:
    """
    Return the start sample index of every repeated symbol in every group.

    Shape: [xi, L]
    """
    repetitions = np.arange(params.symbols_per_group)[:, None]
    groups = np.arange(params.preamble_length)[None, :]
    return groups * params.group_samples + repetitions * params.fft_size


def toa_principal_from_q_indices(
    q_indices: np.ndarray | float,
    params: DerivedParameters,
    M2: int,
) -> np.ndarray:
    """
    Map 2-D FFT delay-bin indices to the paper's principal-value ToA domain.

    Eq. (6) with the long CP example yields:
        D* = -N/M2 * q               if q < M2/2
        D* = -N/M2 * (q - M2)        otherwise

    This representation lives on the signed principal interval [-N/2, N/2].
    """
    q_indices = np.asarray(q_indices, dtype=float)
    wrapped_q = np.where(q_indices < M2 / 2.0, q_indices, q_indices - M2)
    return -(params.fft_size / M2) * wrapped_q


def wrap_delay_samples(delay_samples: np.ndarray | float, span_samples: float) -> np.ndarray:
    """
    Wrap delays into the physical estimation domain [0, span).
    """
    delay_samples = np.asarray(delay_samples, dtype=float)
    return np.mod(delay_samples, span_samples)


def circular_delay_difference_samples(
    estimated_delay_samples: np.ndarray | float,
    true_delay_samples: np.ndarray | float,
    span_samples: float,
) -> np.ndarray:
    """
    Return the minimum wrapped delay difference on a circular delay domain.

    This is the correct way to compare delays when the estimator works on a
    wrapped domain such as [0, Ncp). It removes artificial large tails caused
    by plain subtraction across the wrap boundary.
    """
    estimated_delay_samples = np.asarray(estimated_delay_samples, dtype=float)
    true_delay_samples = np.asarray(true_delay_samples, dtype=float)
    return np.mod(
        estimated_delay_samples - true_delay_samples + span_samples / 2.0,
        span_samples,
    ) - span_samples / 2.0


def parameter_summary(params: DerivedParameters) -> dict:
    """Return a concise parameter summary for validation output."""
    return {
        "fft_size": params.fft_size,
        "sample_rate_hz": params.sample_rate_hz,
        "cp_samples": params.cp_samples,
        "cp_duration_us": 1e6 * params.cp_duration_s_derived,
        "useful_symbol_duration_us": 1e6 * params.useful_symbol_duration_s,
        "group_samples": params.group_samples,
        "group_duration_ms": 1e3 * params.group_duration_s,
        "basic_unit_duration_ms": 1e3 * params.basic_unit_duration_s,
        "preamble_duration_ms": 1e3 * params.preamble_duration_s,
    }
