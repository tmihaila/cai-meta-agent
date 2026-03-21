import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from negmas.sao import (
    SAOMechanism,
    MiCRONegotiator,
    ToughNegotiator,
    TopFractionNegotiator,
    FirstOfferOrientedTBNegotiator,
)
from negmas.gb.negotiators.modular import BOANegotiator
from negmas.gb.components import GACTime, GSmithFrequencyModel, GTimeDependentOffering

from src.domain_loader import list_domains, load_domain
from src.agents import register_agents
from src.hybrid_agent import HybridAgent

STUBBORN_OPPONENTS = {
    "MiCRONegotiator": MiCRONegotiator,
    "ToughNegotiator": ToughNegotiator,
    "TopFractionNegotiator": TopFractionNegotiator,
    "FirstOfferOrientedTBNegotiator": FirstOfferOrientedTBNegotiator,
    "BOA_Hardliner": lambda: BOANegotiator(
        offering=GTimeDependentOffering(e=0.05),
        acceptance=GACTime(t=0.99),
        model=GSmithFrequencyModel(),
    ),
}
register_agents(STUBBORN_OPPONENTS)

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
N_STEPS = 100
ALPHA = 0.7


def compute_score(au, ou):
    return ALPHA * au + (1 - ALPHA) * au * ou


def run_single(issues, ufun_agent, ufun_opp, opp_name, n_steps):
    session = SAOMechanism(issues=issues, n_steps=n_steps)
    agent = HybridAgent(name="hybrid")
    factory = STUBBORN_OPPONENTS[opp_name]
    opponent = factory() if callable(factory) and isinstance(factory, type) else factory()

    session.add(agent, ufun=ufun_agent)
    session.add(opponent, ufun=ufun_opp)

    result = session.run()
    agreement = result.agreement
    au = float(ufun_agent(agreement)) if agreement else float(ufun_agent.reserved_value)
    ou = float(ufun_opp(agreement)) if agreement else float(ufun_opp.reserved_value)

    return {
        "agent_utility": au,
        "opponent_utility": ou,
        "nash": au * ou,
        "score": compute_score(au, ou),
        "agreement": agreement is not None,
        "rounds": result.step,
        "final_e": agent._e,
        "detected_stubborn": agent._is_opponent_stubborn(),
    }


def run_benchmark(max_domains=None):
    domain_paths = list_domains(DOMAINS_DIR)
    if max_domains:
        domain_paths = domain_paths[:max_domains]

    opponents = list(STUBBORN_OPPONENTS.keys())
    total = len(domain_paths) * 2 * len(opponents)
    results = defaultdict(list)
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)

        for role in ("A", "B"):
            ufun_agent = ufun_a if role == "A" else ufun_b
            ufun_opp = ufun_b if role == "A" else ufun_a

            for opp_name in opponents:
                try:
                    row = run_single(issues, ufun_agent, ufun_opp, opp_name, N_STEPS)
                except Exception as e:
                    print(f"ERROR {domain_id} vs {opp_name} role={role}: {e}", file=sys.stderr)
                    continue

                results["domain_id"].append(domain_id)
                results["role"].append(role)
                results["opponent"].append(opp_name)
                results["score"].append(row["score"])
                results["utility"].append(row["agent_utility"])
                results["opp_utility"].append(row["opponent_utility"])
                results["nash"].append(row["nash"])
                results["agreement"].append(row["agreement"])
                results["rounds"].append(row["rounds"])
                results["final_e"].append(row["final_e"])
                results["detected_stubborn"].append(row["detected_stubborn"])

                done += 1
                if done % 20 == 0:
                    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv("data/benchmark_stubborn.csv", index=False)
    print_report(df)
    return df


def print_report(df):
    sep = "=" * 70
    print(f"\n{sep}")
    print("STUBBORN OPPONENT BENCHMARK")
    print(sep)
    print(f"Total matchups: {len(df)}")
    print(f"{'Metric':<30} {'Mean':>8} {'Std':>8}")
    print("-" * 48)
    print(f"{'Score':<30} {df['score'].mean():>8.4f} {df['score'].std():>8.4f}")
    print(f"{'Utility':<30} {df['utility'].mean():>8.4f} {df['utility'].std():>8.4f}")
    print(f"{'Nash product':<30} {df['nash'].mean():>8.4f} {df['nash'].std():>8.4f}")
    print(f"{'Agreement rate':<30} {df['agreement'].mean():>8.2%}")
    print(f"{'Stubborn detected rate':<30} {df['detected_stubborn'].mean():>8.2%}")
    print(f"{'Avg final e':<30} {df['final_e'].mean():>8.4f}")

    print(f"\n{sep}")
    print("PER-OPPONENT BREAKDOWN")
    print(sep)
    print(f"{'Opponent':<35} {'Score':>7} {'Util':>7} {'Nash':>7} {'Agr%':>6} {'Avg e':>7} {'Det%':>6}")
    print("-" * 77)
    for opp_name, group in df.groupby("opponent"):
        print(f"{opp_name:<35} "
              f"{group['score'].mean():>7.4f} "
              f"{group['utility'].mean():>7.4f} "
              f"{group['nash'].mean():>7.4f} "
              f"{group['agreement'].mean():>5.1%} "
              f"{group['final_e'].mean():>7.3f} "
              f"{group['detected_stubborn'].mean():>5.1%}")

    agreed = df[df["agreement"]]
    disagreed = df[~df["agreement"]]
    print(f"\n{sep}")
    print("AGREEMENT vs DISAGREEMENT")
    print(sep)
    print(f"  Agreed ({len(agreed)}):     avg utility = {agreed['utility'].mean():.4f}" if len(agreed) else "  Agreed (0)")
    print(f"  Disagreed ({len(disagreed)}): avg utility = {disagreed['utility'].mean():.4f}" if len(disagreed) else "  Disagreed (0)")
    print(f"\nResults saved to data/benchmark_stubborn.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark HybridAgent against stubborn opponents")
    parser.add_argument("--max-domains", type=int, default=None, help="Limit number of domains (for quick tests)")
    args = parser.parse_args()
    run_benchmark(max_domains=args.max_domains)
