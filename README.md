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

---

## Results

The simulation reproduces the main observations reported in the paper:

* Detection probability > 99%
* False alarm probability ≪ 0.1%
* ToA estimation error concentrated within **[-3, 3] μs**

---

## Notes

* The implementation follows the NPRACH design and receiver structure described in the paper.
* Some implementation details not explicitly specified are handled using documented engineering approximations.
* Minor differences from the original results may occur due to modeling choices and randomness.

---

## Disclaimer

This repository is an independent reproduction of the referenced paper.

* All theoretical concepts belong to the original authors
* This implementation is for academic and research purposes only

---

## Reference

```bibtex
@article{lin2016random,
  title={Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems},
  author={Lin, Xingqin and Adhikary, Ansuman and Wang, Y.-P. Eric},
  journal={IEEE Wireless Communications Letters},
  volume={5},
  number={6},
  pages={640--643},
  year={2016}
}
```
