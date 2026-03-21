import numpy as np
from collections import defaultdict


class FrequencyOpponentModel:
    def __init__(self, issues):
        self._issues = issues
        self._offer_count = 0
        self._value_freq = {
            issue.name: defaultdict(int) for issue in issues
        }

    def update(self, offer):
        if offer is None:
            return
        self._offer_count += 1
        for i, issue in enumerate(self._issues):
            self._value_freq[issue.name][offer[i]] += 1

    def estimate_utility(self, offer):
        """Estimate opponent's utility for an offer based on observed value frequencies.

        Values the opponent proposes more often are assumed to be preferred.
        Each issue is weighted by inverse variance (see _estimate_weights).
        """
        if self._offer_count < 2:
            return 0.5
        weights = self._estimate_weights()
        total = 0.0
        for i, issue in enumerate(self._issues):
            freq = self._value_freq[issue.name]
            total_freq = sum(freq.values())
            val_score = freq.get(offer[i], 0) / total_freq if total_freq > 0 else 0.0
            total += weights[i] * val_score
        return total

    def _estimate_weights(self):
        """Estimate issue weights via inverse-variance of opponent's value choices.

        Low variance means the opponent consistently picks the same values,
        implying that issue is important to them.
        """
        variances = []
        for issue in self._issues:
            freq = self._value_freq[issue.name]
            total = sum(freq.values())
            if total == 0:
                variances.append(1.0)
                continue
            probs = np.array([freq[v] / total for v in issue.values])
            variances.append(float(probs.var()) + 1e-12)
        inv_var = [1.0 / v for v in variances]
        s = sum(inv_var)
        return [w / s for w in inv_var]

    @property
    def is_ready(self):
        return self._offer_count >= 3
