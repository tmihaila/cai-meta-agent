import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from negmas.sao import SAOMechanism

from src.domain_loader import list_domains, load_domain
from src.agents import make_agent, register_agents
from src.hybrid_agent import Group2_Negotiator

from negmas.sao import (
    AspirationNegotiator,
    BoulwareTBNegotiator,
    NaiveTitForTatNegotiator,
    TimeBasedConcedingNegotiator,
    MiCRONegotiator,
    NiceNegotiator,
)

TOURNAMENT_AGENTS = {
    "HybridAgent": lambda: Group2_Negotiator(name="hybrid"),
    "AspirationNegotiator": AspirationNegotiator,
    "BoulwareTBNegotiator": BoulwareTBNegotiator,
    "NaiveTitForTatNegotiator": NaiveTitForTatNegotiator,
    "TimeBasedConcedingNegotiator": TimeBasedConcedingNegotiator,
    "MiCRONegotiator": MiCRONegotiator,
    "NiceNegotiator": NiceNegotiator,
}

register_agents(TOURNAMENT_AGENTS)

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
N_STEPS = 100


def make_tournament_agent(name):
    factory = TOURNAMENT_AGENTS[name]
    if callable(factory) and isinstance(factory, type):
        return factory()
    return factory()


def run_match(issues, ufun_a, ufun_b, name_a, name_b, n_steps):
    session = SAOMechanism(issues=issues, n_steps=n_steps)
    agent_a = make_tournament_agent(name_a)
    agent_b = make_tournament_agent(name_b)

    session.add(agent_a, ufun=ufun_a)
    session.add(agent_b, ufun=ufun_b)

    result = session.run()
    agreement = result.agreement
    ua = float(ufun_a(agreement)) if agreement else float(ufun_a.reserved_value)
    ub = float(ufun_b(agreement)) if agreement else float(ufun_b.reserved_value)

    hybrid_e_a = agent_a._e if isinstance(agent_a, Group2_Negotiator) else None
    hybrid_e_b = agent_b._e if isinstance(agent_b, Group2_Negotiator) else None

    return {
        "utility_a": ua,
        "utility_b": ub,
        "nash": ua * ub,
        "agreement": agreement is not None,
        "hybrid_e_a": hybrid_e_a,
        "hybrid_e_b": hybrid_e_b,
    }


def run_tournament():
    print("starting tournament")
    domain_paths = list_domains(DOMAINS_DIR)
    agent_names = list(TOURNAMENT_AGENTS.keys())
    pairs = [(a, b) for a in agent_names for b in agent_names if a != b]

    total = len(domain_paths) * 2 * len(pairs)
    results = defaultdict(list)
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)

        for role_swap in [False, True]:
            u1 = ufun_a if not role_swap else ufun_b
            u2 = ufun_b if not role_swap else ufun_a

            for name_a, name_b in pairs:
                try:
                    row = run_match(issues, u1, u2, name_a, name_b, N_STEPS)
                except Exception as e:
                    print(f"ERROR {domain_id} {name_a} vs {name_b}: {e}", file=sys.stderr)
                    continue

                results["domain_id"].append(domain_id)
                results["agent_a"].append(name_a)
                results["agent_b"].append(name_b)
                results["utility_a"].append(row["utility_a"])
                results["utility_b"].append(row["utility_b"])
                results["nash"].append(row["nash"])
                results["agreement"].append(row["agreement"])
                results["hybrid_e_a"].append(row["hybrid_e_a"])
                results["hybrid_e_b"].append(row["hybrid_e_b"])

                done += 1
                if done % 200 == 0:
                    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv("data/tournament.csv", index=False)
    print_results(df, agent_names)
    return df


def print_results(df, agent_names):
    sep = "=" * 70

    agent_stats = {}
    for name in agent_names:
        as_a = df[df["agent_a"] == name]
        as_b = df[df["agent_b"] == name]
        utilities = list(as_a["utility_a"]) + list(as_b["utility_b"])
        nash_vals = list(as_a["nash"]) + list(as_b["nash"])
        agreements = list(as_a["agreement"]) + list(as_b["agreement"])

        agent_stats[name] = {
            "utility": np.mean(utilities) if utilities else 0,
            "nash": np.mean(nash_vals) if nash_vals else 0,
            "agreement": np.mean(agreements) if agreements else 0,
            "n": len(utilities),
        }

    ranked = sorted(agent_stats.items(), key=lambda x: x[1]["utility"], reverse=True)

    print(f"\n{sep}")
    print("TOURNAMENT RESULTS — RANKED BY UTILITY")
    print(sep)
    print(f"{'Rank':<5} {'Agent':<30} {'Utility':>8} {'Nash':>8} {'Agree%':>8} {'Games':>6}")
    print("-" * 68)
    for rank, (name, s) in enumerate(ranked, 1):
        tag = " <--" if name == "HybridAgent" else ""
        print(f"{rank:<5} {name:<30} {s['utility']:>8.4f} {s['nash']:>8.4f} {s['agreement']:>7.1%} {s['n']:>6}{tag}")

    print(f"\n{sep}")
    print("HEAD-TO-HEAD: HybridAgent vs each opponent")
    print(sep)
    print(f"{'Opponent':<30} {'Hybrid U':>9} {'Opp U':>9} {'Nash':>8} {'Agree%':>8} {'Avg e':>7}")
    print("-" * 74)
    for opp in agent_names:
        if opp == "HybridAgent":
            continue
        h_as_a = df[(df["agent_a"] == "HybridAgent") & (df["agent_b"] == opp)]
        h_as_b = df[(df["agent_b"] == "HybridAgent") & (df["agent_a"] == opp)]
        h_utils = list(h_as_a["utility_a"]) + list(h_as_b["utility_b"])
        o_utils = list(h_as_a["utility_b"]) + list(h_as_b["utility_a"])
        nash_vals = list(h_as_a["nash"]) + list(h_as_b["nash"])
        agr = list(h_as_a["agreement"]) + list(h_as_b["agreement"])
        e_vals = list(h_as_a["hybrid_e_a"].dropna()) + list(h_as_b["hybrid_e_b"].dropna())
        avg_e = np.mean(e_vals) if e_vals else float("nan")
        print(f"{opp:<30} {np.mean(h_utils):>9.4f} {np.mean(o_utils):>9.4f} "
              f"{np.mean(nash_vals):>8.4f} {np.mean(agr):>7.1%} {avg_e:>7.3f}")

    print(f"\nResults saved to data/tournament.csv")


if __name__ == "__main__":
    run_tournament()
