# Reproduction of "Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems"

This repository implements a reproducible Python NPRACH simulator for:

Xingqin Lin, Ansuman Adhikary, Y.-P. Eric Wang, "Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems," IEEE Wireless Communications Letters, vol. 5, no. 6, 2016.

The project now supports two simulation modes:

- `waveform`: the default paper-reference path. It follows the paper's Section V link-level flow:
  1. map the preamble to an OFDM resource grid
  2. IFFT and CP insertion
  3. explicit upsampling and TX filtering
  4. waveform-domain channel plus AWGN
  5. RX filtering and downsampling
  6. CP removal, FFT, hopped-subcarrier extraction
  7. Eq. (4)-(6) joint ToA / CFO estimation and threshold detection
- `fast`: the existing Eq. (2)-style symbol-domain acceleration path. It remains useful for debugging and benchmarking, but it is no longer the default paper-reproduction mode.

## Paper Settings Implemented

- System bandwidth: `180 kHz`
- Subcarrier spacing: `3.75 kHz`
- Symbol group: `1 CP + 5 symbols`
- Basic unit: `4 symbol groups`
- Preamble lengths: `L in {8, 32, 128}`
- Constant transmitted symbol: `u[m] = 1`
- NPRACH band: `12 subcarriers`
- CP length: `266.7 us`
- Residual CFO: uniform in `[-50, 50] Hz`
- Frequency drift: uniform in `[-22.5, 22.5] Hz/s`
- Timing uncertainty: uniform in `[0, CP)`
- Doppler spread: `1 Hz`
- Antenna configuration: `1 Tx, 2 Rx`
- Target SNRs:
  - `14.25 dB` for `L = 8`
  - `4.25 dB` for `L = 32`
  - `-5.75 dB` for `L = 128`
- Monte Carlo defaults:
  - `10,000` detection trials
  - `100,000` false alarm trials

## Traceability of the Estimator Equations

- Eq. (1): [channel.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/channel.py>)
  - `apply_channel_to_waveform`
- Eq. (2): [channel.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/channel.py>)
  - `synthesize_received_symbols_batch`
- Eq. (3): [channel.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/channel.py>)
  - `b_term`
- Eq. (4) and Eq. (5): [receiver.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/receiver.py>)
  - `direct_joint_search`
- Eq. (6): [receiver.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/receiver.py>)
  - `fft_joint_search_batch`

## Explicit Assumptions and Approximations

The paper does not fully specify several implementation details that a runnable simulator needs. Every non-paper detail below is explicit on purpose.

1. FFT size and discrete-time sample rate:
   The paper gives `180 kHz` and `3.75 kHz` but not the discrete FFT size. This code uses:
   - `N = 64`
   - `Fs = N * 3.75 kHz = 240 kHz`

2. Outer-layer pseudo-random hopping:
   The letter explains pseudo-random hopping but does not publish the exact standardized sequence. The simulator therefore uses a fixed-seed deterministic pseudo-random block offset.

3. Waveform-domain front end:
   The paper includes explicit TX upsampling/filtering and RX filtering/downsampling, but does not specify the exact filters. The simulator uses the following documented engineering approximation:
   - oversampling factor: `8`
   - upsampled sample rate: `1.92 MHz`
   - FIR type: linear-phase low-pass interpolation / anti-alias filter
   - taps: `65`
   - window: `Kaiser(beta=8.0)`
   - cutoff: `0.45 * Fs = 108 kHz`
   - implementation: SciPy `resample_poly` with the designed FIR

   The FIR is not claimed to be the hidden 3GPP implementation. It is a transparent anti-imaging / anti-alias choice to bring the simulation flow closer to the paper.

4. Timing uncertainty in waveform mode:
   Timing uncertainty is sampled on the oversampled grid. With oversampling factor `8`, the physical delay resolution in the waveform front end is `1 / 1.92 MHz`.

5. Typical Urban channel:
   Table I lists `Typical Urban`, but the letter does not provide the exact tap set used in the simulator. This project therefore uses a documented "Typical Urban-like" compact tapped-delay approximation:
   - tap delays: `0.0, 0.5, 1.5, 2.5, 3.5, 5.0 us`
   - tap powers: `0, -1.5, -3, -6, -9, -12 dB`
   - taps normalized to unit average power
   - per-tap Doppler shifts drawn in `[-1, 1] Hz`

   This is explicitly labeled as an approximation, not an exact standardized profile.

6. Two receive branches:
   The receiver combines the two branches noncoherently by summing `|J_g|^2` across receive branches. This preserves the structure of Eq. (4).

7. ToA reporting:
   The estimator still uses the paper's Eq. (6) signed delay mapping internally, then wraps the estimate into `[0, Ncp)` for physical-delay reporting. ToA error is computed using circular delay distance, not plain subtraction, which avoids wraparound artifacts.

8. Waveform-mode SNR normalization:
   In waveform mode, AWGN is added at the receiver input. The simulator measures the receiver front-end noise gain empirically once per case and scales the receiver-input noise variance so that the extracted-symbol noise variance matches the target SNR domain used by Eq. (4)-(6).

9. Fast path:
   The fast mode remains an Eq. (2)-style approximation. For `typical_urban`, it uses the per-group frequency response of the same documented tapped-delay model so it can be benchmarked against the waveform reference path.

## File Overview

- [config.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/config.py>): paper constants, search grids, wrapped-delay helpers, waveform defaults
- [hopping.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/hopping.py>): inner-layer and outer-layer hopping
- [waveform.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/waveform.py>): OFDM resource grid, baseband waveform, front-end FIR design, TX interpolation, preamble reference bank
- [channel.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/channel.py>): fast symbol-domain model, waveform-domain channel, urban-like tapped-delay approximation, receiver-input noise generation
- [receiver.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/receiver.py>): RX filtering/downsampling, CP removal, FFT extraction, direct search, 2-D FFT search
- [detector.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/detector.py>): threshold calibration, independent false alarm evaluation, mode-aware detection experiments
- [simulate_false_alarm.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/simulate_false_alarm.py>): threshold calibration and false alarm evaluation
- [simulate_detection.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/simulate_detection.py>): validation suite plus detection / misdetection experiment
- [simulate_toa_cdf.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/simulate_toa_cdf.py>): ToA CDF generation
- [plots.py](</D:/reproduce_paper_Random Access Preamble Design and Detection for 3GPP NB-IoT/plots.py>): figure generation with fixed ToA axis `[-3, 3] us`

## Running the Experiments

The scripts create mode-specific result bundles automatically under:

- `results/waveform/`
- `results/fast/`

### Paper-reference waveform mode

```bash
python simulate_false_alarm.py --mode waveform --calibration-iterations 100000 --evaluation-iterations 100000 --batch-size 16
python simulate_detection.py --mode waveform --iterations 10000 --threshold-calibration-iterations 100000 --threshold-evaluation-iterations 100000 --batch-size 16
python simulate_toa_cdf.py --mode waveform --iterations 10000 --threshold-calibration-iterations 100000 --threshold-evaluation-iterations 100000 --batch-size 16
python plots.py
```

### Fast-path benchmarking mode

```bash
python simulate_false_alarm.py --mode fast --calibration-iterations 100000 --evaluation-iterations 100000 --batch-size 32
python simulate_detection.py --mode fast --iterations 10000 --threshold-calibration-iterations 100000 --threshold-evaluation-iterations 100000 --batch-size 32
python simulate_toa_cdf.py --mode fast --iterations 10000 --threshold-calibration-iterations 100000 --threshold-evaluation-iterations 100000 --batch-size 32
python plots.py
```

### Recommended smoke tests

```bash
python simulate_false_alarm.py --mode waveform --calibration-iterations 64 --evaluation-iterations 64 --batch-size 4
python simulate_detection.py --mode waveform --iterations 16 --threshold-calibration-iterations 64 --threshold-evaluation-iterations 64 --batch-size 4
python simulate_toa_cdf.py --mode waveform --iterations 32 --threshold-calibration-iterations 64 --threshold-evaluation-iterations 64 --batch-size 4
```

## Validation Workflow Embedded in the Code

The detection script prints:

1. parameter derivation
2. hopping validation
3. waveform front-end summary
4. waveform-vs-fast extracted-symbol comparison on a controlled unit-gain case
5. direct Eq. (4)-(5) search result
6. 2-D FFT Eq. (6) result and waveform-vs-fast estimator agreement

The same threshold calibration and wrapped-delay ToA error logic are then reused by the Monte Carlo experiments.

## Output Files

After running one mode, the following files are produced under `results/<mode>/`:

- `false_alarm_results.json`
- `detection_results.json`
- `toa_cdf_results.npz`
- `toa_cdf_summary.json`
- `detection_false_alarm.png`
- `toa_cdf.png`

The JSON / NPZ metadata record the actual run configuration and trial counts for
that bundle. The scripts default to paper-scale counts, but any generated bundle
should be interpreted using the counts stored in its own metadata rather than by
assuming the defaults were used.

## Reproducibility Notes

- All random draws use explicit NumPy generator seeds.
- Threshold calibration and false alarm evaluation use independent noise-only data sets.
- Threshold files now include run-configuration metadata and are only reused when `mode`, `channel_model`, `m1`, `m2`, and `num_rx` match.
- `plots.py` validates that the detection, false alarm, and ToA files come from one compatible mode-specific result bundle before generating figures.
- The waveform and fast modes share the same estimator and the same ToA wrapping logic, so their differences are concentrated in the front-end and channel modeling rather than in the detector core.
