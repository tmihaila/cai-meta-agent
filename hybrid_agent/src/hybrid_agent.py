import numpy as np
import torch
from pathlib import Path

from negmas.sao import SAONegotiator, ResponseType

from src.features import extract_all_features, FEATURE_COLUMNS
from src.opponent_model import FrequencyOpponentModel
from src.hybrid_nn import ConcessionNet

MODEL_DIR = Path("models")
DEFAULT_E = 0.5  # e < 1 = boulware, e = 1 = linear, e > 1 = conceder


class HybridAgent(SAONegotiator):
    def __init__(self, *args, model_dir: Path = MODEL_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_dir = model_dir
        self._e = DEFAULT_E
        self._opponent_model = None
        self._best_opponent_util = -1.0
        self._first_opponent_util = None
        self._prev_opponent_util = -1.0
        self._opponent_concessions = 0
        self._opponent_offers_count = 0
        self._adapt_count = 0
        self._all_outcomes = None
        self._best_opponent_offer = None
        self._load_model()

    def _load_model(self):
        self._nn_model = ConcessionNet()
        state_dict = torch.load(self._model_dir / "concession_net.pt", weights_only=True)
        self._nn_model.load_state_dict(state_dict)
        self._nn_model.eval()
        self._scaler_mean = np.load(self._model_dir / "scaler_mean.npy")
        self._scaler_scale = np.load(self._model_dir / "scaler_scale.npy")

    def _predict_e(self, first_offer):
        """Predict the concession exponent e using the NN model.

        Uses domain, profile, and first-offer features to select an optimal
        concession speed. Returns e clamped to [0.1, 2.0].
        """
        features = extract_all_features(self.nmi.issues, self.ufun, first_offer)
        x = np.array([features[c] for c in FEATURE_COLUMNS], dtype=np.float32)
        x = (x - self._scaler_mean) / (self._scaler_scale + 1e-12)  # z-score normalization
        with torch.no_grad():
            e = self._nn_model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).item()
        return max(0.1, min(e, 2.0))

    def _target_utility(self, t):
        """Time-dependent concession curve: starts at 1.0 and decays toward rv.

        The decay shape is controlled by self._e:
        e < 1 = boulware (hold firm), e = 1 = linear, e > 1 = conceder.
        """
        rv = float(self.ufun.reserved_value)
        return rv + (1.0 - rv) * (1.0 - t ** (1.0 / self._e))

    def _ensure_outcomes(self):
        """Precomputes and caches all possible outcomes."""
        if self._all_outcomes is None:
            self._all_outcomes = list(self.nmi.outcome_space.enumerate_or_sample(max_cardinality=10000))

    def _opponent_concession_rate(self):
        if self._opponent_offers_count < 2:
            return 0.5
        return self._opponent_concessions / self._opponent_offers_count

    def _is_opponent_stubborn(self):
        """Detect non-conceding opponents using three signals:

        1. Low concession rate (<10% of offers improve on previous)
        2. Low total movement (<0.08 utility gain since first offer)
        3. Low per-round gain (<0.003/round), catches micro-conceders like MiCRO
        """
        if self._opponent_offers_count < 5:
            return False
        if self._opponent_concession_rate() < 0.1:  # rarely concedes
            return True
        if self._opponent_offers_count >= 6:
            total_movement = self._best_opponent_util - self._first_opponent_util
            if total_movement < 0.08:  # barely moved in absolute terms
                return True
            gain_per_round = total_movement / self._opponent_offers_count
            if gain_per_round < 0.003:  # micro-conceders (e.g. MiCRO)
                return True
        return False

    def _select_offer(self, target_util, t=0.0):
        """Pick the best outcome near target_util, balancing own and estimated
        opponent utility. Against stubborn opponents in late game, the opponent
        weight increases to make offers more likely to be accepted.
        """
        self._ensure_outcomes()
        best_offer = None
        best_score = -float("inf")

        stubborn = self._is_opponent_stubborn()
        opp_weight = 0.3
        if stubborn:
            if t > 0.8:
                opp_weight = 1.0
            elif t > 0.6:
                opp_weight = 0.6

        for outcome in self._all_outcomes:
            u = float(self.ufun(outcome))
            if u < target_util * 0.9:  # skip outcomes too far below target
                continue
            if self._opponent_model is not None and self._opponent_model.is_ready:
                opp_u = self._opponent_model.estimate_utility(outcome)
                score = u + opp_weight * opp_u  # balance own + estimated opponent utility
            else:
                score = u
            if score > best_score:
                best_score = score
                best_offer = outcome

        if best_offer is None:
            # Fallback: pick outcome closest to target if nothing passed the filter
            utilities = [(o, float(self.ufun(o))) for o in self._all_outcomes]
            utilities.sort(key=lambda x: abs(x[1] - target_util))
            best_offer = utilities[0][0] if utilities else self.nmi.random_outcomes(1)[0]

        return best_offer

    def _ensure_opponent_model(self):
        if self._opponent_model is None:
            self._opponent_model = FrequencyOpponentModel(self.nmi.issues)

    def propose(self, state):
        t = state.relative_time
        self._ensure_opponent_model()

        if self._first_opponent_util is None:
            # First move: open with our maximum-utility outcome
            self._ensure_outcomes()
            return max(self._all_outcomes, key=lambda o: float(self.ufun(o)))

        if t > 0.98 and self._best_opponent_offer is not None:
            # Last resort: play back the best offer we received
            return self._best_opponent_offer

        target = self._target_utility(t)
        if self._is_opponent_stubborn() and t > 0.8:
            # Drop target aggressively to find mutually acceptable outcomes
            rv = float(self.ufun.reserved_value)
            target = max(rv + 0.05 * (1.0 - rv), target * 0.8)
        return self._select_offer(target, t)

    def respond(self, state, source=None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER

        self._ensure_opponent_model()
        self._opponent_model.update(offer)

        t = state.relative_time
        offer_util = float(self.ufun(offer))
        self._opponent_offers_count += 1
        if self._first_opponent_util is None:
            self._first_opponent_util = offer_util
            self._e = self._predict_e(offer)  # set concession speed from NN on first offer
        if offer_util > self._best_opponent_util:
            self._best_opponent_util = offer_util
            self._best_opponent_offer = offer
        if offer_util > self._prev_opponent_util + 0.01:  # count as concession if > 1% improvement
            self._opponent_concessions += 1
        self._prev_opponent_util = offer_util

        self._adapt_e(t)

        target = self._target_utility(t)
        rv = float(self.ufun.reserved_value)

        if offer_util >= target:
            return ResponseType.ACCEPT_OFFER

        stubborn = self._is_opponent_stubborn()

        # Tiered acceptance against stubborn opponents
        if stubborn:
            if t > 0.8:
                acceptable_floor = rv + 0.1 * (target - rv)  # just above rv
                if offer_util >= acceptable_floor:
                    return ResponseType.ACCEPT_OFFER
            if t > 0.9 and offer_util >= rv:  # accept anything above BATNA
                return ResponseType.ACCEPT_OFFER

        # General deadline fallbacks
        if t > 0.97 and offer_util >= rv:
            return ResponseType.ACCEPT_OFFER

        if t > 0.9 and self._best_opponent_util > rv and offer_util >= self._best_opponent_util * 0.98:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _adapt_e(self, t):
        """Adjust concession exponent e based on observed opponent behavior.

        Fires at most once per time threshold (0.4, 0.7, 0.85).
        Increases e (concede faster) against tough opponents,
        decreases e (hold firmer) against generous ones.
        """
        if self._opponent_offers_count < 3:
            return
        thresholds = [0.4, 0.7, 0.85]
        if self._adapt_count >= len(thresholds) or t < thresholds[self._adapt_count]:
            return
        self._adapt_count += 1
        concession_rate = self._opponent_concessions / self._opponent_offers_count
        if concession_rate < 0.1:     # hardliner -> concede faster
            self._e = min(self._e * 2.0, 2.0)
        elif concession_rate < 0.15:  # tough -> concede somewhat faster
            self._e = min(self._e * 1.5, 1.5)
        elif concession_rate > 0.5:   # generous -> hold firmer
            self._e = max(self._e * 0.7, 0.05)
