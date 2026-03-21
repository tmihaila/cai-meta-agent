# Meta-Agent for Automated Negotiation — Implementation Plan

## Overview

Build a meta-agent that, given a negotiation domain and the opponent's first offer, selects the best-performing agent strategy from a portfolio of 5 NegMAS negotiators. Inspired by the algorithm selection approach in the AAAI 2013 paper (Ilany & Gal).

---

## Phase 1: Agent Portfolio

We use 5 built-in NegMAS SAO negotiators as our portfolio:

| # | Agent | Strategy |
|---|-------|----------|
| 1 | `AspirationNegotiator` | Time-based concession with aspiration curve |
| 2 | `BoulwareTBNegotiator` | Tough — concedes very slowly, holds out |
| 3 | `NaiveTitForTatNegotiator` | Reactive — mirrors opponent's concessions |
| 4 | `TimeBasedConcedingNegotiator` | Conceder — gives in faster over time |
| 5 | `BOANegotiator(GTimeDependentOffering(e=0.2), GACNext, GSmithFrequencyModel)` | BOA-based with Boulware offering, next-offer acceptance, and frequency-based opponent model |

---

## Phase 2: Dataset Generation

### 2.1 Simulation Setup

For each of the **50 domains**:
- Each domain defines `issues`, `ufun_a`, `ufun_b`
- For every **(agent, opponent)** pair where agent ∈ PORTFOLIO and opponent ∈ OPPONENTS:
  - Run negotiation with agent playing as **Agent A** (using `ufun_a`) and opponent playing as **Agent B** (using `ufun_b`)
  - Run negotiation with agent playing as **Agent B** (using `ufun_b`) and opponent playing as **Agent A** (using `ufun_a`)
  - No repeats — diversity comes from the larger opponent pool

#### Opponent Pool (diverse, beyond portfolio)

We use a **broader set of ~10 opponents** to ensure the meta-agent generalizes to unknown opponents:

| # | Opponent | Notes |
|---|----------|-------|
| 1–5 | The 5 portfolio agents | Self-play |
| 6 | `LinearTBNegotiator` | Linear concession rate |
| 7 | `ConcederTBNegotiator` | Fast conceder |
| 8 | `BOANegotiator(e=0.05)` | Very tough BOA (hardliner) |
| 9 | `BOANegotiator(e=1.0)` | Very conceding BOA |
| 10 | `BOANegotiator(e=0.5, GACTime(t=0.9))` | Moderate BOA with time-based acceptance |

**Total rows ≈ 50 domains × 5 agents × 10 opponents × 2 roles = 5,000 rows**

Negotiation parameters:
- Protocol: `SAOMechanism`
- Steps: **100** (configurable)
- If no agreement: utility = `reserved_value` for both agents

> **Key design decision**: At inference time the opponent is unknown. The NN does **not** receive `opponent_type` as input — it learns to predict the expected score **averaged across possible opponents**, conditioned only on domain features + profile + first offer utility. The first offer utility acts as an implicit signal about the opponent's strategy.

### 2.2 Data Capture

For each negotiation, we intercept the opponent's **first offer** and record:

| Column | Description |
|--------|-------------|
| `domain_id` | Domain identifier (0–49) |
| `agent_type` | Portfolio agent class name |
| `opponent_type` | Opponent agent class name |
| `role` | Which profile the agent plays (`A` or `B`) |
| `first_offer_utility` | Utility of opponent's first offer **to our agent** |
| `agent_utility` | Final utility achieved by our agent |
| `opponent_utility` | Final utility achieved by opponent |
| `nash_product` | Nash product of both agents' final utilities (`u_agent × u_opponent`) |
| `agreement_reached` | Boolean — did they agree? |
| `rounds` | Number of rounds before agreement/deadline |
| Domain features | See §3 below |
| Profile features | See §3 below |

### 2.3 Implementation: `generate_dataset.py`

```
for domain_file in domains_python/:
    load issues, ufun_a, ufun_b
    for agent_cls in PORTFOLIO:
        for opponent_cls in OPPONENTS:
            for role in ['A', 'B']:
                session = SAOMechanism(issues, n_steps=100)
                # assign agent & opponent with correct ufuns based on role
                # use a wrapper/hook to capture opponent's first offer
                result = session.run()
                # extract utilities, nash product, features → append row
    save to CSV
```

To capture the opponent's first offer, we can:
- Use a **thin wrapper negotiator** around the actual agent that records the first incoming offer in `respond()`, OR
- Inspect the mechanism's negotiation history after running (`session.history`)

---

## Phase 3: Feature Engineering

### 3.1 Domain Features (common knowledge — same for both roles)

| Feature | Description |
|---------|-------------|
| `num_issues` | Number of issues in the domain |
| `total_values` | Total number of discrete values across all issues |
| `avg_values_per_issue` | Mean number of values per issue |
| `max_values_per_issue` | Max values in any single issue |
| `min_values_per_issue` | Min values in any single issue |
| `log_outcome_space_size` | log₂ of the product of all issue cardinalities |

### 3.2 Profile Features (private info — depends on which ufun we play)

| Feature | Description |
|---------|-------------|
| `reserved_value` | Agent's reservation value |
| `max_weight` | Highest issue weight |
| `min_weight` | Lowest issue weight |
| `weight_std` | Std deviation of issue weights |
| `weight_entropy` | Shannon entropy of the weight distribution (normalized) |
| `avg_value_std` | Mean std deviation of utility values per issue — how polarized preferences are |
| `weight_concentration` | HHI (Herfindahl–Hirschman Index) of normalized weights — how concentrated preferences are |

### 3.3 First Offer Features

| Feature | Description |
|---------|-------------|
| `first_offer_utility` | Utility of the opponent's first offer to us (scalar in [0, 1]) |

This is the simplest and most informative representation. It collapses the full categorical offer into a single number that tells us how aggressive or generous the opponent's opening is.

> **Rationale**: A high first-offer utility to us likely means a cooperative opponent; a low value suggests a tough/boulware opponent. This is a strong signal for which of our agents will perform best.

### 3.4 Total Input Features

**6 (domain) + 7 (profile) + 1 (first offer) = 14 features**

All features are continuous and can be standardized (z-score normalization) before feeding into the NN.

---

## Phase 4: Target Variable

### Composite Score

For each row, we compute a **composite score** that blends own utility and Nash product:

```
score = α × agent_utility + (1 − α) × nash_product
```

Where `nash_product = agent_utility × opponent_utility` (already in [0, 1] if utilities are in [0, 1]).

Default: **α = 0.7** (prioritize own utility but reward cooperative outcomes via Nash product).

---

## Phase 5: Neural Network

### 5.1 Architecture

- **Input**: 14 features (standardized)
- **Hidden layer 1**: 32 neurons, ReLU activation
- **Output**: 5 neurons (one per portfolio agent), linear activation (regression)
- **Loss**: MSE on predicted score vs actual score

#### Why 1 layer is likely sufficient
- Low-dimensional input (14 features)
- Relatively smooth relationships (domain complexity → agent performance)
- The paper found no significant difference between CART, NN, and linear regression — suggesting the signal is not deeply nonlinear
- With ~5,000 samples and 14 features, 1 layer avoids overfitting

#### When to consider 2 layers
- If validation loss plateaus significantly above training loss with 1 layer
- If we increase the dataset (more domains/repeats)
- A second layer with 16 neurons could help capture feature interactions

**Recommendation: start with 1 hidden layer (32 neurons). Try 2 layers (32 → 16) as a comparison.**

### 5.2 Training Setup

- Framework: **PyTorch** (lightweight, easy to customize)
- Train/val/test split: **70/15/15** — split by domain (not by row!) to test generalization to unseen domains
- Optimizer: Adam, lr=1e-3
- Early stopping on validation loss (patience ~20 epochs)
- Batch size: 64

### 5.3 Inference (Meta-Agent at Runtime)

```
Given a new domain + our profile:
  1. Compute domain features (6) + profile features (7)
  2. Receive opponent's first offer → compute first_offer_utility (1)
  3. Feed 14 features into trained NN → get 5 predicted scores
  4. Select agent with highest predicted score
  5. Delegate the rest of the negotiation to that agent
```

> **Note**: The meta-agent must handle the first round itself (make an initial proposal to receive the opponent's first offer). Following the paper, it proposes its maximum-utility outcome as the opening bid.

---

## Phase 6: Implementation Files

| File | Purpose |
|------|---------|
| `generate_dataset.py` | Run all simulations, extract features, save CSV |
| `features.py` | Feature extraction functions (domain, profile, first offer) |
| `train_nn.py` | Load CSV, train NN, save model |
| `meta_agent.py` | The runtime meta-agent negotiator (NegMAS-compatible) |
| `evaluate.py` | Compare meta-agent vs individual agents vs AvgBest baseline |
| `requirements.txt` | Dependencies: negmas, torch, pandas, numpy, scikit-learn |

---

## Decisions (resolved)

1. **Agent portfolio**: 4 built-in + 1 BOA agent ✅
2. **Negotiation steps**: 100 ✅
3. **Opponents**: No repeats — instead use ~10 diverse opponents (portfolio + extra agents) ✅
4. **Social welfare**: Nash product (`u_agent × u_opponent`) ✅
5. **Score blending**: `score = 0.7 × utility + 0.3 × nash_product` ✅
6. **Train/test split**: By domain ✅
7. **First offer**: Meta-agent proposes max-utility outcome, then observes opponent's first offer ✅
8. **Unknown opponents**: NN does not receive opponent type as input; it averages over opponents implicitly ✅
