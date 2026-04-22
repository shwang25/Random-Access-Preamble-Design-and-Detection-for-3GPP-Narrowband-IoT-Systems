# Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems

This repository provides a **reproduction and implementation study** of the paper:

> **"Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems"**  
> Xingqin Lin, Ansuman Adhikary, Y.-P. Eric Wang  
> IEEE Wireless Communications Letters, 2016

---

## Overview

This project reproduces the core simulation framework of the paper, including:

- NPRACH single-tone preamble generation  
- Frequency hopping design  
- Joint ToA / residual CFO estimation  
- Detection and false alarm evaluation  
- ToA estimation error CDF under different coverage classes  

Two simulation modes are supported:

- `waveform`: waveform-domain link-level simulation (closer to the paper)  
- `fast`: symbol-domain approximation (for debugging and benchmarking)  

---

## Repository Structure

```text
├── config.py                  # Simulation parameters
├── hopping.py                 # NPRACH hopping pattern
├── waveform.py                # Waveform generation
├── channel.py                 # Channel modeling
├── receiver.py                # Receiver processing
├── detector.py                # Detection logic
├── simulate_false_alarm.py    # False alarm simulation
├── simulate_detection.py      # Detection simulation
├── simulate_toa_cdf.py        # ToA CDF simulation
├── plots.py                   # Plot generation
````

---

## How to Run

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run (Waveform Mode)

```bash
python simulate_false_alarm.py --mode waveform
python simulate_detection.py --mode waveform
python simulate_toa_cdf.py --mode waveform
python plots.py
```

### Run (Fast Mode)

```bash
python simulate_false_alarm.py --mode fast
python simulate_detection.py --mode fast
python simulate_toa_cdf.py --mode fast
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

After running the scripts, the following files are produced under `results/`:

- `false_alarm_results.json`
- `detection_results.json`
- `toa_cdf_results.npz`
- `toa_cdf_summary.json`
- `detection_false_alarm.png`
- `toa_cdf.png`

## Reproducibility Notes

- All random draws use explicit NumPy generator seeds.
- Threshold calibration and false alarm evaluation use independent noise-only data sets.
- The waveform and fast modes share the same estimator and the same ToA wrapping logic, so their differences are concentrated in the front-end and channel modeling rather than in the detector core.
