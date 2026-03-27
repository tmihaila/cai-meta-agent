import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

from negmas.sao import (
    SAOMechanism,
    MiCRONegotiator,
    NiceNegotiator,
    RandomNegotiator,
    ToughNegotiator,
    SimpleTitForTatNegotiator,
    TopFractionNegotiator,
    FirstOfferOrientedTBNegotiator,
)

from src.domain_loader import list_domains, load_domain
from src.agents import OPPONENT_NAMES, OPPONENTS, register_agents, make_agent
from src.group2_negotiator import Group2_Negotiator

UNSEEN_OPPONENTS = {
    "Group2_Negotiator": Group2_Negotiator(name="self_hybrid"),
    "MiCRONegotiator": MiCRONegotiator,
    "NiceNegotiator": NiceNegotiator,
    "RandomNegotiator": RandomNegotiator,
    "ToughNegotiator": ToughNegotiator,
    "SimpleTitForTatNegotiator": SimpleTitForTatNegotiator,
    "TopFractionNegotiator": TopFractionNegotiator,
    "FirstOfferOrientedTBNegotiator": FirstOfferOrientedTBNegotiator,
}
register_agents(UNSEEN_OPPONENTS)

ALL_EVAL_OPPONENTS = {**OPPONENTS, **UNSEEN_OPPONENTS}

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
ALPHA = 0.7
N_STEPS = 1


def compute_score(au, ou):
    return ALPHA * au + (1 - ALPHA) * au * ou

def enumerate_outcomes(issues):
    from itertools import product
    values = [issue.values for issue in issues]
    return list(product(*values))


def compute_pareto_frontier(outcomes, ufun_a, ufun_b):
    pareto = []
    for o in outcomes:
        ua, ub = float(ufun_a(o)), float(ufun_b(o))
        dominated = False
        for o2 in outcomes:
            ua2, ub2 = float(ufun_a(o2)), float(ufun_b(o2))
            if (ua2 >= ua and ub2 >= ub) and (ua2 > ua or ub2 > ub):
                dominated = True
                break
        if not dominated:
            pareto.append((ua, ub))
    return pareto


def distance_to_pareto(point, pareto_front):
    if point is None:
        return None
    au, ou = point
    return min(((au - p[0]) ** 2 + (ou - p[1]) ** 2) ** 0.5 for p in pareto_front)


def compute_kalai_point(outcomes, ufun_a, ufun_b):
    max_ua = max(float(ufun_a(o)) for o in outcomes)
    max_ub = max(float(ufun_b(o)) for o in outcomes)

    best = None
    best_val = -1

    for o in outcomes:
        ua, ub = float(ufun_a(o)), float(ufun_b(o))
        val = min(ua / max_ua if max_ua > 0 else 0,
                  ub / max_ub if max_ub > 0 else 0)
        if val > best_val:
            best_val = val
            best = (ua, ub)

    return best


def run_hybrid_negotiation(issues, ufun_agent, ufun_opp, opp_name, n_steps,
                          pareto_front, kalai_point):
    session = SAOMechanism(issues=issues, n_steps=n_steps)
    agent = Group2_Negotiator(name="hybrid")
    if (opp_name == "Group2_Negotiator"):
        opponent = Group2_Negotiator(name="opponent_hybrid")
    else:
        opponent = make_agent(opp_name)

    session.add(agent, ufun=ufun_agent)
    session.add(opponent, ufun=ufun_opp)

    result = session.run()
    agreement = result.agreement
    au = float(ufun_agent(agreement)) if agreement else float(ufun_agent.reserved_value)
    ou = float(ufun_opp(agreement)) if agreement else float(ufun_opp.reserved_value)

    point = (au, ou) if agreement else None

    pareto_dist = distance_to_pareto(point, pareto_front) if point else None

    kalai_dist = None
    if point:
        kalai_dist = ((point[0] - kalai_point[0]) ** 2 +
                    (point[1] - kalai_point[1]) ** 2) ** 0.5

    social_welfare = au + ou

    return {
        "agent_utility": au,
        "opponent_utility": ou,
        "nash": au * ou,
        "score": compute_score(au, ou),
        "agreement": agreement is not None,
        "rounds": result.step,
        "predicted_e": agent._e,
        "social_welfare": social_welfare,
        "pareto_dist": pareto_dist,
        "kalai_dist": kalai_dist,
    }


def evaluate(test_domains=None, seen_only=False, unseen_only=False):
    domain_paths = list_domains(DOMAINS_DIR)

    if test_domains:
        domain_paths = [p for p in domain_paths if p.stem in test_domains]

    if seen_only:
        opponents = list(OPPONENTS.keys())
    elif unseen_only:
        opponents = list(UNSEEN_OPPONENTS.keys())
    else:
        opponents = list(ALL_EVAL_OPPONENTS.keys())

    total = len(domain_paths) * 2 * len(opponents)
    results = defaultdict(list)
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)
        outcomes = enumerate_outcomes(issues)
        pareto_front = compute_pareto_frontier(outcomes, ufun_a, ufun_b)
        kalai_point = compute_kalai_point(outcomes, ufun_a, ufun_b)

        for role in ("A", "B"):
            ufun_agent = ufun_a if role == "A" else ufun_b
            ufun_opp = ufun_b if role == "A" else ufun_a

            for opp_name in opponents:
                try:
                    row = run_hybrid_negotiation(
                        issues, ufun_agent, ufun_opp, opp_name, N_STEPS,
                        pareto_front, kalai_point
                    )   
                except Exception as e:
                    print(f"ERROR {domain_id} vs {opp_name} role={role}: {e}", file=sys.stderr)
                    continue

                is_seen = opp_name in OPPONENTS
                results["domain_id"].append(domain_id)
                results["role"].append(role)
                results["opponent"].append(opp_name)
                results["opponent_seen"].append(is_seen)
                results["score"].append(row["score"])
                results["utility"].append(row["agent_utility"])
                results["opp_utility"].append(row["opponent_utility"])
                results["nash"].append(row["nash"])
                results["agreement"].append(row["agreement"])
                results["rounds"].append(row["rounds"])
                results["predicted_e"].append(row["predicted_e"])
                results["social_welfare"].append(row["social_welfare"])
                results["pareto_dist"].append(row["pareto_dist"])
                results["kalai_dist"].append(row["kalai_dist"])

                done += 1
                if done % 50 == 0:
                    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv("data/evaluation.csv", index=False)
    print_statistics(df)
    return df


def print_statistics(df: pd.DataFrame):
    sep = "=" * 130

    print("(* = unseen during training)")
    print(f"\n{sep}")
    print("(Group2_Negotiator vs Group2_Negotiator)")
    print(sep)

    df_self = df[df["opponent"] == "Group2_Negotiator"]

    if not df_self.empty:
        print(f"Total self-play matchups: {len(df_self)}")
        print(f"{'Metric':<25} {'Mean':>8}")
        print("-" * 35)
        print(f"{'Agreement rate':<25} {df_self['agreement'].mean():>8.2%}")
        print(f"{'Utility':<25} {df_self['utility'].mean():>8.4f}")
        print(f"{'Nash':<25} {df_self['nash'].mean():>8.4f}")
        print(f"{'Social welfare':<25} {df_self['social_welfare'].mean():>8.4f}")
        print(f"{'Pareto distance':<25} {df_self['pareto_dist'].mean():>8.4f}")
        print(f"{'Kalai distance':<25} {df_self['kalai_dist'].mean():>8.4f}")
    else:
        print("No self-play data found.")

    print(f"\n{sep}")
    print("Group2_Negotiator RESULTS")
    print(sep)
    print(f"Total matchups: {len(df)}")
    print(f"{'Metric':<25} {'Mean':>8} {'Std':>8}")
    print("-" * 43)
    print(f"{'Score':<25} {df['score'].mean():>8.4f} {df['score'].std():>8.4f}")
    print(f"{'Utility':<25} {df['utility'].mean():>8.4f} {df['utility'].std():>8.4f}")
    print(f"{'Nash':<25} {df['nash'].mean():>8.4f} {df['nash'].std():>8.4f}")
    print(f"{'Agreement rate':<25} {df['agreement'].mean():>8.2%}")
    print(f"{'Avg predicted e':<25} {df['predicted_e'].mean():>8.4f}")
    print(f"{'Social welfare':<25} {df['social_welfare'].mean():>8.4f} {df['social_welfare'].std():>8.4f}")
    print(f"{'Pareto distance':<25} {df['pareto_dist'].mean():>8.4f} {df['pareto_dist'].std():>8.4f}")
    print(f"{'Kalai distance':<25} {df['kalai_dist'].mean():>8.4f} {df['kalai_dist'].std():>8.4f}")

    if "opponent_seen" in df.columns and df["opponent_seen"].nunique() > 1:
        print(f"\n{sep}")
        print("SEEN vs UNSEEN OPPONENTS")
        print(sep)
        for label, subset in [("Seen", df[df["opponent_seen"]]), ("Unseen", df[~df["opponent_seen"]])]:
            if subset.empty:
                continue
            print(f"\n  [{label} opponents] (n={len(subset)})")
            print(f"  {'Metric':<25} {'Mean':>8}")
            print(f"  {'-' * 35}")
            print(f"  {'Score':<25} {subset['score'].mean():>8.4f}")
            print(f"  {'Utility':<25} {subset['utility'].mean():>8.4f}")
            print(f"  {'Nash':<25} {subset['nash'].mean():>8.4f}")
            print(f"  {'Agreement rate':<25} {subset['agreement'].mean():>8.2%}")

    print(f"\n{sep}")
    print("PER-OPPONENT BREAKDOWN")
    print(sep)
    print(f"{'Opponent':<35} {'Score':>7} {'Util':>7} {'Nash':>7} {'Social Welfare':>17} {'Pareto Dist':>14} {'Kalai Dist':>14} {'Agr%':>6} {'Avg e':>7}")
    print("-" * 130)
    for opp_name, group in df.groupby("opponent"):
        seen_tag = "" if group["opponent_seen"].iloc[0] else " *"
        print(f"{opp_name + seen_tag:<35} "
              f"{group['score'].mean():>7.4f} "
              f"{group['utility'].mean():>7.4f} "
              f"{group['nash'].mean():>7.4f} "
              f"{group['social_welfare'].mean():>16.4f} "
              f"{group['pareto_dist'].mean():>14.4f} "
              f"{group['kalai_dist'].mean():>14.4f} "
              f"{group['agreement'].mean():>7.1%} "
              f"{group['predicted_e'].mean():>7.3f}")
        
    print("(* = unseen during training)")
    print(f"\nResults saved to data/evaluation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-domains", nargs="+", default=None)
    parser.add_argument("--seen-only", action="store_true")
    parser.add_argument("--unseen-only", action="store_true")
    args = parser.parse_args()
    evaluate(test_domains=args.test_domains, seen_only=args.seen_only, unseen_only=args.unseen_only)
