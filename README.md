# Random Access Preamble Design and Detection for 3GPP Narrowband IoT Systems

This repository provides a **waveform-domain reproduction and implementation study** of the paper:

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

The implementation focuses on the waveform-domain paper-reference path.

---

## Repository Structure

```text
├── config.py                  # Simulation parameters
├── hopping.py                 # NPRACH hopping pattern
├── waveform.py                # Waveform generation
├── channel.py                 # Waveform-domain channel modeling
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

```bash
python simulate_false_alarm.py
python simulate_detection.py
python simulate_toa_cdf.py
python plots.py
```

---

## Notes

- This repository implements the waveform-domain paper-like simulation chain only
- Minor differences from the paper may remain because some details not fully specified in the letter are handled with explicit engineering approximations
- The code reproduces the core NPRACH mechanism and main performance trends, but does not claim exact hidden 3GPP or paper-internal simulator equivalence

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
