# HybridAgent — Architecture & Strategy

## Overview

HybridAgent is a time-dependent negotiation agent built on `negmas.sao.SAONegotiator`. It combines three layers:

1. **Neural network** — predicts the initial concession speed per negotiation
2. **Frequency-based opponent model** — estimates what the opponent values
3. **Runtime adaptation** — detects stubborn opponents and adjusts strategy mid-session

The core idea: instead of using a fixed concession strategy, the agent adapts both *before* and *during* the negotiation based on domain characteristics and opponent behavior.

---

## Concession Curve

The agent's target utility at time `t ∈ [0, 1]` follows:

```
target(t) = rv + (1 - rv) * (1 - t^(1/e))
```

- `rv` — reserved value (BATNA, minimum acceptable utility)
- `e` — concession exponent:
  - `e < 1` → **Boulware**: hold firm, concede only near the deadline
  - `e = 1` → **Linear**: constant concession rate
  - `e > 1` → **Conceder**: concede early, stabilize later

The `e` parameter is **not fixed** — it is predicted by the neural network on the first offer and then adapted at runtime.

---

## Neural Network (`ConcessionNet`)

A small feedforward network that predicts the optimal `e` for a given negotiation:

- **Architecture**: `Linear(14, 32) → ReLU → Linear(32, 1) → Softplus`
- **Input**: 14 z-score normalized features (see below)
- **Output**: `e`, clamped to `[0.1, 2.0]`
- **Trigger**: runs once, when the first opponent offer is received

### Training Features (14 total)

| Category | Feature | Description |
|---|---|---|
| **Domain** | `num_issues` | Number of negotiation issues |
| | `total_values` | Sum of values across all issues |
| | `avg_values_per_issue` | Mean cardinality per issue |
| | `max_values_per_issue` | Largest issue cardinality |
| | `min_values_per_issue` | Smallest issue cardinality |
| | `log_outcome_space_size` | log2 of total outcome combinations |
| **Profile** | `reserved_value` | Agent's BATNA |
| | `max_weight` | Highest issue weight |
| | `min_weight` | Lowest issue weight |
| | `weight_std` | Std dev of issue weights |
| | `weight_entropy` | Normalized entropy of weights (1.0 = uniform) |
| | `avg_value_std` | Average std of value mappings per issue |
| | `weight_concentration` | Herfindahl index of weights (1.0 = one dominant issue) |
| **Opponent** | `first_offer_utility` | Utility of the opponent's first offer to us |

---

## Opponent Model (`FrequencyOpponentModel`)

Estimates the opponent's utility function by tracking how often they propose each value:

- **Issue weights**: estimated via **inverse-variance** — if the opponent consistently picks the same value for an issue, that issue is likely important to them
- **Value scores**: frequency of each value normalized by total offers
- **Ready after**: 3 offers received

Used in offer selection to pick outcomes that are attractive to both sides.

---

## Offer Selection (`_select_offer`)

For each candidate outcome above `target * 0.9`:

```
score = own_utility + opp_weight * estimated_opponent_utility
```

- Default `opp_weight = 0.3`
- Against stubborn opponents: increases to `0.6` (t > 0.6) or `1.0` (t > 0.8)

This biases late-game offers toward what the opponent is likely to accept.

**Fallback**: if no outcome passes the filter, picks the one closest to target utility.

---

## Stubborn Opponent Detection (`_is_opponent_stubborn`)

Activates after 5 offers. Uses three independent signals:

1. **Low concession rate** — fewer than 10% of offers improve on the previous one
2. **Low total movement** — best offer utility minus first offer utility < 0.08
3. **Low per-round gain** — total movement / rounds < 0.003 (catches micro-conceders like MiCRONegotiator)

Any one signal is sufficient to flag the opponent as stubborn.

---

## Runtime `e` Adaptation (`_adapt_e`)

At three time checkpoints (t = 0.4, 0.7, 0.85), the agent re-evaluates opponent behavior and adjusts `e`:

| Opponent concession rate | Action |
|---|---|
| < 10% | `e *= 2.0` (cap 2.0) — concede much faster |
| < 15% | `e *= 1.5` (cap 1.5) — concede somewhat faster |
| > 50% | `e *= 0.7` (floor 0.05) — hold firmer |

Each checkpoint fires **at most once** (ratcheted).

---

## Bidding Strategy (`propose`)

The bidding strategy determines what the agent offers each round. It follows a phased approach:

### Phase 1 — Opening (first round)

The agent always opens with its **maximum-utility outcome**. This is a standard anchoring tactic — start at the best possible position and concede from there. No opponent information is available yet.

### Phase 2 — Concession-Curve Bidding (normal rounds)

The agent computes a target utility using the concession curve (`_target_utility`) and then selects the best outcome near that target via `_select_offer`. The offer selection balances two objectives:

- **Own utility**: the outcome should be close to or above the target
- **Opponent utility**: using the frequency-based opponent model, the agent estimates how much the opponent would like each outcome and factors it in

The scoring formula is `score = own_util + opp_weight * opp_util`. This means the agent doesn't just pick the best outcome for itself — it actively seeks **win-win outcomes** that the opponent is more likely to accept.

### Phase 3 — Stubborn Adjustment (t > 0.8, stubborn detected)

If the opponent is flagged as stubborn and we're past 80% of the negotiation:

- The target utility drops to `max(rv + 5% * (1 - rv), target * 0.8)`, making the agent willing to propose outcomes closer to its BATNA
- The opponent weight in offer scoring jumps to `0.6` (t > 0.6) or `1.0` (t > 0.8), meaning late-game offers are heavily optimized for what the opponent wants

This combination makes it much more likely that a stubborn opponent will accept, while still keeping us above the reserved value.

### Phase 4 — Best-Offer Replay (t > 0.98)

In the final 2% of the negotiation, the agent stops computing new offers and instead **plays back the best offer it ever received** from the opponent. This signals willingness to agree on the opponent's own terms and maximizes the chance of a last-second agreement.

---

## Acceptance Strategy (`respond`)

The acceptance strategy determines whether to accept or reject an incoming offer. It uses a tiered system evaluated in order, combining the concession curve target with deadline-aware fallbacks.

### Tier 1 — Standard Acceptance

```
if offer_util >= target(t): ACCEPT
```

The baseline condition: accept any offer that meets or exceeds our current target utility from the concession curve. This is the primary acceptance mechanism for cooperative opponents.

### Tier 2 — Stubborn Opponent Acceptance (early deadline)

When the opponent is detected as stubborn:

- **t > 0.8**: accept if the offer is above `rv + 10% * (target - rv)`. This floor is only slightly above the reserved value — the agent is willing to take a low-utility deal rather than risk disagreement.
- **t > 0.9**: accept **anything at or above the reserved value**. At this point, any deal is better than the BATNA.

These tiers activate much earlier than normal deadline acceptance, giving the agent more rounds to reach agreement with a non-conceding opponent.

### Tier 3 — General Deadline Fallbacks

These apply regardless of stubborn detection:

- **t > 0.97**: accept anything at or above `rv`. A last-resort safety net to avoid disagreement.
- **t > 0.9**: accept if the current offer is within 2% of the **best offer we've ever received** from the opponent. The logic: if the opponent has shown they can offer us X, and the current offer is close to X, we should take it rather than gamble on getting something better in the few remaining rounds.

### Decision Flow Summary

```
offer received
  │
  ├─ offer >= target(t)?  ──────────────────── ACCEPT
  │
  ├─ stubborn AND t > 0.8?
  │    └─ offer >= rv + 10%(target - rv)?  ─── ACCEPT
  │
  ├─ stubborn AND t > 0.9?
  │    └─ offer >= rv?  ──────────────────────  ACCEPT
  │
  ├─ t > 0.97 AND offer >= rv?  ──────────────  ACCEPT
  │
  ├─ t > 0.9 AND offer ≈ best_seen (98%)?  ── ACCEPT
  │
  └─ else  ───────────────────────────────────  REJECT
```

---

## Key Files

| File | Purpose |
|---|---|
| `hybrid_agent.py` | Main agent class with all negotiation logic |
| `hybrid_nn.py` | ConcessionNet neural network definition |
| `features.py` | Feature extraction for NN input (domain, profile, first offer) |
| `opponent_model.py` | Frequency-based opponent utility estimation |
| `agents.py` | Agent registry and factory (portfolio + evaluation opponents) |
| `simulation.py` | Single negotiation runner for dataset generation |
| `domain_loader.py` | Loads negotiation domains from Python files |

---

## Scripts

| Script | Purpose |
|---|---|
| `generate_dataset.py` | Runs negotiations to create training data |
| `train_nn.py` | Trains ConcessionNet on generated dataset |
| `evaluate.py` | Evaluates HybridAgent against seen + unseen opponents |
| `tournament.py` | Full round-robin tournament between all agent types |
| `benchmark_stubborn.py` | Targeted benchmark against non-conceding opponents |
