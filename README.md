# Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems

This repository provides a **reproduction and implementation study** of the paper:

> **"Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems"**  
> Xingqin Lin, Ansuman Adhikary, Y.-P. Eric Wang  
> IEEE Wireless Communications Letters, 2016

---

## Overview

This project reproduces the main NPRACH simulation framework of the paper, including:

- Single-tone NPRACH preamble generation  
- Frequency hopping design  
- Joint ToA / residual CFO estimation  
- Detection and false alarm evaluation  
- ToA estimation error CDF under different coverage classes  

Two simulation modes are included:

- `waveform` for the paper-like waveform-domain link-level simulation  
- `fast` for a lighter symbol-domain approximation  

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
```

---

## How to Run

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Run

Waveform mode:

```bash
python simulate_false_alarm.py --mode waveform
python simulate_detection.py --mode waveform
python simulate_toa_cdf.py --mode waveform
python plots.py --mode waveform
```

Fast mode:

```bash
python simulate_false_alarm.py --mode fast
python simulate_detection.py --mode fast
python simulate_toa_cdf.py --mode fast
python plots.py --mode fast
```

---

## Notes

- `waveform` mode is the main paper-reference path and is closer to the link-level procedure described in the paper
- `fast` mode is useful for efficient evaluation and benchmarking
- Minor differences from the paper may remain due to explicit engineering approximations in implementation details not fully specified in the letter

---

## Disclaimer

This repository is an independent reproduction of the referenced paper.

- All methods and theoretical concepts belong to the original authors
- This implementation is provided for academic and research purposes only

---

## Reference

```bibtex
@article{lin2016random,
  title={Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems},
  author={Lin, Xingqin and Adhikary, Ansuman and Wang, Y.-P. Eric},
  journal={IEEE Wireless Communications Letters},
  year={2016}
}
```