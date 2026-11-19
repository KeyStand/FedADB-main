# FedADB: Class Anchor-Driven Dual-Branch Federated Learning for Mitigating Forgetting

This repository contains the official implementation of **FedADB**, a federated learning framework designed to mitigate catastrophic forgetting under heterogeneous data distributions.

## Paper

**FedADB: Class Anchor-Driven Dual-Branch Federated Learning for Mitigating Forgetting**  
Zhenyan Liu, Hua Zhang, Haoran Gao, Qi Li, Hongliang Zhu, Huiyu Zhou, Zongliang Shen, Yanxin Xu, and Jiahui Wang.  
Proceedings of the 34th ACM International Conference on Multimedia (MM '26).

DOI: [10.1145/3767308.3835615](https://doi.org/10.1145/3767308.3835615)

## Overview

FedADB addresses two common challenges in federated learning under non-IID data:

- **Catastrophic forgetting** caused by local training on heterogeneous client data
- **Missing-class supervision** when some classes are absent on individual clients

The framework introduces two main components:

1. **Class Anchor Generation (CAG)**: the server synthesizes class anchors from the global model in input space.
2. **Dual-Branch Cooperative Training (DBCT)**: each client trains with an anchor-guided global branch and a local calibration branch to balance global consistency and local adaptability.

## Key Features

- Lightweight class-anchor supervision without relying on proxy datasets
- Dual-branch local training for better balance between global alignment and local optimization
- Support for common FL scenarios such as **cross-silo** and **cross-device** settings
- Compatible with multiple datasets and backbones used in the paper

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

### Main Dependencies

- `python`
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `wandb`

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KeyStand/FedADB-main.git
cd FedADB-main
pip install -r requirements.txt
```

## Quick Start

Run FedADB with the default configuration:

```bash
python main.py --config_path ./config/fedadb.json
```

Run the FedAvg baseline:

```bash
python main.py --config_path ./config/fedavg.json
```

You can also override configuration values from the command line. For example:

```bash
python main.py --config_path ./config/fedadb.json --dataset_name cifar10 --n_clients 20 --device cuda:0
```

## Configuration

Configuration files are stored in the `config/` directory.

Common options include:

- `data_setups`
  - dataset name
  - number of clients
  - batch size
  - partition strategy
- `train_setups`
  - algorithm name
  - number of rounds
  - client sampling ratio
  - local epochs
  - optimizer and scheduler settings
- `wandb_setups`
  - project name
  - group name
  - run name

See `config/fedadb.json` for a complete example.

## Repository Structure

```text
FedADB/
├── algorithms/              # Federated learning algorithms
│   ├── fedavg/               # FedAvg baseline
│   └── fedadb/               # FedADB implementation
├── config/                   # Training configuration files
├── train_tools/              # Model and data processing utilities
├── main.py                   # Entry point
├── utils.py                  # Shared helper functions
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Notes

- The code is organized for research and reproduction purposes.
- Datasets are not included in the repository and should be prepared separately.
- If you use this code in your research, please cite the paper above.

## Citation

```bibtex
@inproceedings{liu2026fedadb,
  title     = {FedADB: Class Anchor-Driven Dual-Branch Federated Learning for Mitigating Forgetting},
  author    = {Liu, Zhenyan and Zhang, Hua and Gao, Haoran and Li, Qi and Zhu, Hongliang and Zhou, Huiyu and Shen, Zongliang and Xu, Yanxin and Wang, Jiahui},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  doi       = {10.1145/3767308.3835615}
}
```