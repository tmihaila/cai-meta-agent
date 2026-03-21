import math
import numpy as np
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


def domain_features(issues) -> dict:
    cardinalities = [len(issue.values) for issue in issues]
    n = len(issues)
    total = sum(cardinalities)
    return {
        "num_issues": n,
        "total_values": total,
        "avg_values_per_issue": total / n,
        "max_values_per_issue": max(cardinalities),
        "min_values_per_issue": min(cardinalities),
        "log_outcome_space_size": sum(math.log2(c) for c in cardinalities),
    }


def profile_features(ufun: LUFun) -> dict:
    weights = np.array(ufun.weights)
    w_norm = weights / weights.sum() if weights.sum() > 0 else weights
    entropy = -np.sum(w_norm * np.log2(w_norm + 1e-12))
    max_entropy = math.log2(len(weights)) if len(weights) > 1 else 1.0

    stds = []
    for vals in ufun.values:
        mapping = vals.mapping if hasattr(vals, "mapping") else vals
        v = np.array(list(mapping.values()) if isinstance(mapping, dict) else list(mapping))
        stds.append(float(v.std()))

    return {
        "reserved_value": ufun.reserved_value,
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min()),
        "weight_std": float(weights.std()),
        "weight_entropy": float(entropy / max_entropy),
        "avg_value_std": float(np.mean(stds)),
        "weight_concentration": float((w_norm ** 2).sum()),
    }


def first_offer_features(offer, ufun: LUFun) -> dict:
    if offer is None:
        return {"first_offer_utility": ufun.reserved_value}
    return {"first_offer_utility": float(ufun(offer))}


def extract_all_features(issues, ufun: LUFun, first_offer=None) -> dict:
    return {
        **domain_features(issues),
        **profile_features(ufun),
        **first_offer_features(first_offer, ufun),
    }


FEATURE_COLUMNS = list(extract_all_features.__code__.co_varnames)[:0] or [
    "num_issues",
    "total_values",
    "avg_values_per_issue",
    "max_values_per_issue",
    "min_values_per_issue",
    "log_outcome_space_size",
    "reserved_value",
    "max_weight",
    "min_weight",
    "weight_std",
    "weight_entropy",
    "avg_value_std",
    "weight_concentration",
    "first_offer_utility",
]
