import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from negmas.sao import (
    MiCRONegotiator,
    NiceNegotiator,
    RandomNegotiator,
    ToughNegotiator,
    SimpleTitForTatNegotiator,
    TopFractionNegotiator,
    FirstOfferOrientedTBNegotiator,
)

from src.domain_loader import list_domains, load_domain
from src.simulation import run_negotiation
from src.agents import PORTFOLIO_NAMES, OPPONENTS, register_agents

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
ALPHA = 0.7
N_STEPS = 100

UNSEEN_OPPONENTS = {
    "MiCRONegotiator": MiCRONegotiator,
    "NiceNegotiator": NiceNegotiator,
    "RandomNegotiator": RandomNegotiator,
    "ToughNegotiator": ToughNegotiator,
    "SimpleTitForTatNegotiator": SimpleTitForTatNegotiator,
    "TopFractionNegotiator": TopFractionNegotiator,
    "FirstOfferOrientedTBNegotiator": FirstOfferOrientedTBNegotiator,
}

ALL_EVAL_OPPONENTS = {**OPPONENTS, **UNSEEN_OPPONENTS}
register_agents(UNSEEN_OPPONENTS)


def compute_score(agent_util, opp_util):
    return ALPHA * agent_util + (1 - ALPHA) * agent_util * opp_util


def evaluate_fixed(test_domains=None, seen_only=False, unseen_only=False):
    domain_paths = list_domains(DOMAINS_DIR)

    if test_domains:
        domain_paths = [p for p in domain_paths if p.stem in test_domains]

    if seen_only:
        opponents = list(OPPONENTS.keys())
    elif unseen_only:
        opponents = list(UNSEEN_OPPONENTS.keys())
    else:
        opponents = list(ALL_EVAL_OPPONENTS.keys())

    total = len(domain_paths) * 2 * len(opponents) * len(PORTFOLIO_NAMES)
    results = defaultdict(list)
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)

        for role in ("A", "B"):
            ufun_agent = ufun_a if role == "A" else ufun_b
            ufun_opp = ufun_b if role == "A" else ufun_a

            for opp_name in opponents:
                is_seen = opp_name in OPPONENTS
                for agent_name in PORTFOLIO_NAMES:
                    try:
                        row = run_negotiation(issues, ufun_agent, ufun_opp, agent_name, opp_name, N_STEPS)
                    except Exception as e:
                        print(f"ERROR {domain_id} {agent_name} vs {opp_name}: {e}", file=sys.stderr)
                        continue
                    au = row["agent_utility"]
                    ou = row["opponent_utility"]
                    results["domain_id"].append(domain_id)
                    results["role"].append(role)
                    results["opponent"].append(opp_name)
                    results["opponent_seen"].append(is_seen)
                    results["agent"].append(agent_name)
                    results["utility"].append(au)
                    results["opp_utility"].append(ou)
                    results["nash"].append(au * ou)
                    results["score"].append(compute_score(au, ou))
                    results["agreement"].append(row["agreement_reached"])

                    done += 1
                    if done % 200 == 0:
                        print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv("data/evaluation_fixed.csv", index=False)
    print_statistics(df)
    return df


def print_statistics(df: pd.DataFrame):
    sep = "=" * 70

    print(f"\n{sep}")
    print("FIXED-AGENT EVALUATION")
    print(sep)
    print(f"Total negotiations: {len(df)}")

    print(f"\n{'Agent':<30} {'Score':>8} {'Utility':>8} {'Nash':>8} {'Agree%':>8}")
    print("-" * 66)
    for agent_name in PORTFOLIO_NAMES:
        sub = df[df["agent"] == agent_name]
        print(f"{agent_name:<30} "
              f"{sub['score'].mean():>8.4f} "
              f"{sub['utility'].mean():>8.4f} "
              f"{sub['nash'].mean():>8.4f} "
              f"{sub['agreement'].mean():>7.1%}")

    if "opponent_seen" in df.columns and df["opponent_seen"].nunique() > 1:
        for label, mask in [("Seen", df["opponent_seen"]), ("Unseen", ~df["opponent_seen"])]:
            subset = df[mask]
            if subset.empty:
                continue
            print(f"\n{sep}")
            print(f"FIXED-AGENT vs {label.upper()} OPPONENTS (n={len(subset)})")
            print(sep)
            print(f"{'Agent':<30} {'Score':>8} {'Utility':>8} {'Nash':>8} {'Agree%':>8}")
            print("-" * 66)
            for agent_name in PORTFOLIO_NAMES:
                sub = subset[subset["agent"] == agent_name]
                print(f"{agent_name:<30} "
                      f"{sub['score'].mean():>8.4f} "
                      f"{sub['utility'].mean():>8.4f} "
                      f"{sub['nash'].mean():>8.4f} "
                      f"{sub['agreement'].mean():>7.1%}")

    print(f"\n{sep}")
    print("PER-OPPONENT BREAKDOWN (fixed agent avg score)")
    print(sep)
    header = f"{'Opponent':<35} " + " ".join(f"{a[:8]:>8}" for a in PORTFOLIO_NAMES)
    print(header)
    print("-" * len(header))
    for opp_name, opp_group in df.groupby("opponent"):
        seen_tag = "" if opp_group["opponent_seen"].iloc[0] else " *"
        vals = []
        for agent_name in PORTFOLIO_NAMES:
            sub = opp_group[opp_group["agent"] == agent_name]
            vals.append(f"{sub['score'].mean():>8.4f}")
        print(f"{opp_name + seen_tag:<35} " + " ".join(vals))
    print("(* = unseen during training)")

    print(f"\nResults saved to data/evaluation_fixed.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-domains", nargs="+", default=None)
    parser.add_argument("--seen-only", action="store_true")
    parser.add_argument("--unseen-only", action="store_true")
    args = parser.parse_args()
    evaluate_fixed(test_domains=args.test_domains, seen_only=args.seen_only, unseen_only=args.unseen_only)
