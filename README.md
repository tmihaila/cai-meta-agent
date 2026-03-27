# HybridAgent

A time-dependent negotiation agent built on [NegMAS](https://github.com/yasserfarouk/negmas). It combines a neural network for predicting concession speed, a frequency-based opponent model, and runtime adaptation against stubborn opponents.

## Project Structure

```
hybrid_agent/
├── src/
│   ├── hybrid_agent.py      # Main agent (negotiation logic)
│   ├── hybrid_nn.py          # ConcessionNet neural network
│   ├── features.py           # Feature extraction for NN input
│   ├── opponent_model.py     # Frequency-based opponent model
│   ├── agents.py             # Agent registry and factory
│   ├── simulation.py         # Single negotiation runner
│   └── domain_loader.py      # Loads domains from Python files
├── models/                    # Trained model weights and scaler
├── data/                      # Generated datasets and results
├── generate_dataset.py        # Create training data
├── train_nn.py                # Train the neural network
├── evaluate.py                # Evaluate against seen + unseen opponents
└── tournament.py              # Round-robin tournament
domains_python/                # Negotiation domain definitions
requirements.txt
```

## Negotiation Domains

The 50 negotiation domains in `domains_python/` were sourced from the [Automated Negotiating Agents Competition (ANAC) 2023](https://web.tuat.ac.jp/~katfuji/ANAC2023/). They were converted from the original XML format into NegMAS-compatible Python modules. During this conversion, the reservation value for both agents was changed from the original 0 to a universal value of 0.4. 

## Prerequisites

- Python 3.10+

## Installation

```bash
pip install -r requirements.txt
```

## Running the Agent

All scripts must be run from the `hybrid_agent/` directory:

```bash
cd hybrid_agent
```

### 1. Generate Training Dataset

Runs negotiations across all domains, roles, opponents, and concession exponent values to build a labeled dataset.

```bash
python generate_dataset.py
```

Output: `data/dataset.csv`

### 2. Train the Neural Network

Trains `ConcessionNet` on the generated dataset using group k-fold cross-validation (grouped by domain).

```bash
python train_nn.py
```

Optional arguments:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 300 | Maximum training epochs |
| `--lr` | 0.001 | Learning rate |
| `--patience` | 20 | Early stopping patience |

Example:

```bash
python train_nn.py --epochs 500 --lr 5e-4 --patience 30
```

Output: `models/concession_net.pt`, `models/scaler_mean.npy`, `models/scaler_scale.npy`, `models/meta.json`

### 3. Evaluate the Agent

Evaluates `HybridAgent` against both seen (training) and unseen opponents across all domains.

```bash
python evaluate.py
```

Optional arguments:

| Flag | Description |
|---|---|
| `--test-domains DOMAIN1 DOMAIN2 ...` | Restrict to specific domains |
| `--seen-only` | Evaluate only against training opponents |
| `--unseen-only` | Evaluate only against unseen opponents |

Example:

```bash
python evaluate.py --unseen-only
```

Output: `data/evaluation.csv`

### 4. Run a Tournament

Full round-robin tournament between `HybridAgent` and several baseline agents.

```bash
python tournament.py
```

Output: `data/tournament.csv`

## Quick Start (End-to-End)

If starting from scratch (no pre-trained model):

```bash
cd hybrid_agent
python generate_dataset.py
python train_nn.py
python evaluate.py
```

If a trained model already exists in `models/`, you can skip straight to evaluation or the tournament:

```bash
cd hybrid_agent
python evaluate.py
python tournament.py
```
