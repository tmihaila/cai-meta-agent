# Hybrid Agent Plan

## Overview

A single custom negotiation agent (`GroupN_Negotiator`) that uses a neural network to tune its concession rate based on domain features and the opponent's first offer, combined with a frequency-based opponent model for Pareto-aware offer generation.

All negotiation logic (offering, acceptance, opponent modeling) is implemented from scratch.

---

## Architecture

```
Round 1:  Propose max-utility offer → observe opponent's first offer
          → NN predicts optimal concession exponent (e)

Round 2+: Time-dependent concession (NN-tuned e)
          + Frequency-based opponent model
          → Pareto-aware offer selection
          → Accept when opponent offer ≥ concession target
```

### Component 1: Offering Strategy (time-dependent concession)

The agent concedes over time using:

```
target_utility(t) = min_util + (1 - min_util) * (1 - t^(1/e))
```

Where:
- `t` = relative time ∈ [0, 1]
- `e` = concession exponent (predicted by NN)
  - `e < 1` → boulware (concede late)
  - `e = 1` → linear
  - `e > 1` → conceder (concede early)
- `min_util` = reservation value (floor)

From all outcomes, select the one closest to `target_utility(t)` that also maximizes estimated opponent utility (Pareto-awareness via opponent model).

### Component 2: Acceptance Strategy

Accept opponent's offer if any of:
1. `opponent_offer_utility ≥ target_utility(t)` — offer is at least as good as what we'd propose
2. `t > 0.98 and opponent_offer_utility ≥ reservation_value` — near deadline, accept anything above reservation
3. `opponent_offer_utility ≥ best_offer_so_far * 0.98 and t > 0.9` — opponent is near their best, close to deadline

### Component 3: Opponent Model (frequency-based)

Track all opponent offers to estimate their preferences:
1. For each issue, count how often each value is proposed
2. Normalize frequencies → estimated value function per issue
3. Estimate issue weights from how much the opponent varies each issue (low variance = high weight)
4. Use estimated opponent ufun to score our candidate offers and prefer Pareto-efficient ones

### Component 4: NN-Based Parameter Prediction

A lightweight neural network predicts the optimal `e` value:
- Input: 14 features (6 domain + 7 profile + 1 first offer utility)
- Output: 1 float (optimal `e`)
- Trained on simulation data where we sweep `e` values and record which one scored best

---

## Training Data Generation

### Simulation Setup

Instead of 5 fixed portfolio agents, use a single time-dependent agent with varying `e`:

- **e values**: {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0} — 9 values
- **Opponents**: 17 total (10 seen + 7 diverse)
  - Seen: AspirationNegotiator, BoulwareTBNegotiator, NaiveTitForTatNegotiator, TimeBasedConcedingNegotiator, BOA_Boulware, LinearTBNegotiator, ConcederTBNegotiator, BOA_Hardliner, BOA_Conceder, BOA_Moderate
  - Unseen: MiCRONegotiator, NiceNegotiator, RandomNegotiator, ToughNegotiator, SimpleTitForTatNegotiator, TopFractionNegotiator, FirstOfferOrientedTBNegotiator
- **Domains**: 50
- **Roles**: A and B

Total negotiations: 50 × 2 × 17 × 9 = **15,300**

### Labeling

For each `(domain, role, opponent)` group (1,700 groups):
1. Compute composite score for each `e`: `score = 0.7 * agent_utility + 0.3 * nash_product`
2. The `e` that produced the highest score is the label: `optimal_e`

Training set: 1,700 rows × (14 features → 1 target)

### Training

- Model: 1 hidden layer (32 neurons), ReLU, single output
- Loss: MSE on `e`
- Validation: GroupKFold by domain
- Standardize features (z-score)
- Early stopping (patience 20)

---

## File Structure

```
src/
├── __init__.py
├── agents.py              — opponent registry (shared with meta_agent)
├── domain_loader.py       — domain loading (shared)
├── features.py            — feature extraction (shared)
├── simulation.py          — negotiation runner (shared)
├── opponent_model.py      — NEW: frequency-based opponent modeling
├── hybrid_agent.py        — NEW: the main GroupN_Negotiator
└── hybrid_nn.py           — NEW: NN model for e prediction

generate_hybrid_dataset.py — NEW: sweep e values, generate training data
train_hybrid_nn.py         — NEW: train NN to predict optimal e
evaluate_hybrid.py         — NEW: benchmark the hybrid agent
```

Existing `meta_agent/` folder is untouched.

---

## Evaluation Plan

### Baselines
- Each fixed `e` value (0.05 to 2.0) as a standalone agent
- Best single `e` across all matchups (equivalent to "always pick one strategy")
- Oracle: per-matchup best `e`

### Metrics
- Composite score (0.7 × utility + 0.3 × nash)
- Raw utility
- Nash product
- Agreement rate
- Distance from Pareto frontier (if computable)

### Splits
- Seen vs unseen opponents
- Per-opponent breakdown
- Per-domain breakdown

---

## Implementation Order

1. `src/opponent_model.py` — frequency-based opponent preference estimator
2. `src/hybrid_agent.py` — the negotiator with time-dependent offering, acceptance, opponent model
3. `generate_hybrid_dataset.py` — sweep e values across domains/opponents
4. `src/hybrid_nn.py` — simple NN (14 → 32 → 1)
5. `train_hybrid_nn.py` — train on generated dataset
6. Integrate NN into `hybrid_agent.py` for first-round e prediction
7. `evaluate_hybrid.py` — full benchmarking
