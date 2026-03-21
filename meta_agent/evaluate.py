import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
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
from src.agents import PORTFOLIO_NAMES, OPPONENT_NAMES, OPPONENTS, register_agents
from src.features import FEATURE_COLUMNS
from src.nn_model import AgentScoreNet, AgentScoreNet2Layer

MODEL_DIR = Path("models")
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


def load_model():
    with open(MODEL_DIR / "meta.json") as f:
        meta = json.load(f)
    model_cls = AgentScoreNet2Layer if meta["model_class"] == "AgentScoreNet2Layer" else AgentScoreNet
    model = model_cls()
    state_dict = torch.load(MODEL_DIR / "agent_score_net.pt", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    scaler_mean = np.load(MODEL_DIR / "scaler_mean.npy")
    scaler_scale = np.load(MODEL_DIR / "scaler_scale.npy")
    return model, scaler_mean, scaler_scale


def predict_best_agent(model, scaler_mean, scaler_scale, features: dict) -> str:
    x = np.array([features[c] for c in FEATURE_COLUMNS], dtype=np.float32)
    x = (x - scaler_mean) / (scaler_scale + 1e-12)
    with torch.no_grad():
        scores = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return PORTFOLIO_NAMES[int(np.argmax(scores))]


def compute_score(agent_util, opp_util):
    return ALPHA * agent_util + (1 - ALPHA) * agent_util * opp_util


def evaluate(test_domains=None, seen_only=False, unseen_only=False):
    model, scaler_mean, scaler_scale = load_model()
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

        for role in ("A", "B"):
            ufun_agent = ufun_a if role == "A" else ufun_b
            ufun_opp = ufun_b if role == "A" else ufun_a

            for opp_name in opponents:
                agent_data = {}
                features = None
                for agent_name in PORTFOLIO_NAMES:
                    try:
                        row = run_negotiation(issues, ufun_agent, ufun_opp, agent_name, opp_name, N_STEPS)
                    except Exception as e:
                        print(f"ERROR {domain_id} {agent_name} vs {opp_name}: {e}", file=sys.stderr)
                        agent_data[agent_name] = {"utility": 0.0, "opp_utility": 0.0, "nash": 0.0, "score": 0.0}
                        continue
                    au = row["agent_utility"]
                    ou = row["opponent_utility"]
                    agent_data[agent_name] = {
                        "utility": au, "opp_utility": ou,
                        "nash": au * ou, "score": compute_score(au, ou),
                    }
                    if features is None:
                        features = {c: row[c] for c in FEATURE_COLUMNS}

                if features is None:
                    continue

                meta_choice = predict_best_agent(model, scaler_mean, scaler_scale, features)
                oracle_choice = max(agent_data, key=lambda a: agent_data[a]["score"])
                avgbest_score = np.mean([d["score"] for d in agent_data.values()])
                is_seen = opp_name in OPPONENTS

                results["domain_id"].append(domain_id)
                results["role"].append(role)
                results["opponent"].append(opp_name)
                results["opponent_seen"].append(is_seen)
                results["meta_agent_choice"].append(meta_choice)
                results["meta_agent_score"].append(agent_data[meta_choice]["score"])
                results["meta_agent_utility"].append(agent_data[meta_choice]["utility"])
                results["meta_agent_nash"].append(agent_data[meta_choice]["nash"])
                results["oracle_choice"].append(oracle_choice)
                results["oracle_score"].append(agent_data[oracle_choice]["score"])
                results["oracle_utility"].append(agent_data[oracle_choice]["utility"])
                results["oracle_nash"].append(agent_data[oracle_choice]["nash"])
                results["avgbest_score"].append(avgbest_score)
                results["meta_agrees_oracle"].append(meta_choice == oracle_choice)
                for agent_name in PORTFOLIO_NAMES:
                    d = agent_data.get(agent_name, {"score": 0.0, "utility": 0.0, "nash": 0.0})
                    results[f"score_{agent_name}"].append(d["score"])
                    results[f"utility_{agent_name}"].append(d["utility"])
                    results[f"nash_{agent_name}"].append(d["nash"])

                done += 1
                if done % 50 == 0:
                    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(results)
    df.to_csv("data/evaluation.csv", index=False)
    print_statistics(df)
    return df


def print_statistics(df: pd.DataFrame):
    sep = "=" * 60

    print(f"\n{sep}")
    print("OVERALL RESULTS")
    print(sep)
    print(f"Total matchups: {len(df)}")
    print(f"{'Strategy':<30} {'Score':>8} {'Utility':>8} {'Nash':>8}")
    print("-" * 58)
    print(f"{'Meta-agent':<30} {df['meta_agent_score'].mean():>8.4f} {df['meta_agent_utility'].mean():>8.4f} {df['meta_agent_nash'].mean():>8.4f}")
    print(f"{'Oracle (upper bound)':<30} {df['oracle_score'].mean():>8.4f} {df['oracle_utility'].mean():>8.4f} {df['oracle_nash'].mean():>8.4f}")
    avg_util = np.mean([df[f'utility_{a}'].mean() for a in PORTFOLIO_NAMES])
    avg_nash = np.mean([df[f'nash_{a}'].mean() for a in PORTFOLIO_NAMES])
    print(f"{'AvgBest (baseline)':<30} {df['avgbest_score'].mean():>8.4f} {avg_util:>8.4f} {avg_nash:>8.4f}")
    for agent_name in PORTFOLIO_NAMES:
        s = df[f"score_{agent_name}"].mean()
        u = df[f"utility_{agent_name}"].mean()
        n = df[f"nash_{agent_name}"].mean()
        print(f"  {agent_name:<28} {s:>8.4f} {u:>8.4f} {n:>8.4f}")
    print(f"Meta agrees w/ oracle:   {df['meta_agrees_oracle'].mean():.2%}")

    meta_beats_avg = (df["meta_agent_score"] > df["avgbest_score"]).mean()
    meta_beats_or_ties_avg = (df["meta_agent_score"] >= df["avgbest_score"]).mean()
    print(f"Meta beats AvgBest:      {meta_beats_avg:.2%}")
    print(f"Meta beats/ties AvgBest: {meta_beats_or_ties_avg:.2%}")

    if "opponent_seen" in df.columns and df["opponent_seen"].nunique() > 1:
        print(f"\n{sep}")
        print("SEEN vs UNSEEN OPPONENTS")
        print(sep)
        for label, subset in [("Seen", df[df["opponent_seen"]]), ("Unseen", df[~df["opponent_seen"]])]:
            if subset.empty:
                continue
            print(f"\n  [{label} opponents] (n={len(subset)})")
            print(f"  {'Strategy':<25} {'Avg Score':>10}")
            print(f"  {'-'*37}")
            print(f"  {'Meta-agent':<25} {subset['meta_agent_score'].mean():>10.4f}")
            print(f"  {'Oracle':<25} {subset['oracle_score'].mean():>10.4f}")
            print(f"  {'AvgBest':<25} {subset['avgbest_score'].mean():>10.4f}")
            print(f"  Meta agrees w/ oracle: {subset['meta_agrees_oracle'].mean():.2%}")

    print(f"\n{sep}")
    print("PER-OPPONENT BREAKDOWN")
    print(sep)
    print(f"{'Opponent':<35} {'Meta':>7} {'Oracle':>7} {'AvgBst':>7} {'Agree%':>7}")
    print("-" * 65)
    for opp_name, group in df.groupby("opponent"):
        seen_tag = "" if group["opponent_seen"].iloc[0] else " *"
        print(f"{opp_name + seen_tag:<35} "
              f"{group['meta_agent_score'].mean():>7.4f} "
              f"{group['oracle_score'].mean():>7.4f} "
              f"{group['avgbest_score'].mean():>7.4f} "
              f"{group['meta_agrees_oracle'].mean():>6.1%}")
    print("(* = unseen during training)")

    print(f"\n{sep}")
    print("META-AGENT SELECTION FREQUENCY")
    print(sep)
    counts = df["meta_agent_choice"].value_counts()
    for agent_name in PORTFOLIO_NAMES:
        n = counts.get(agent_name, 0)
        print(f"  {agent_name:<30} {n:>5} ({100*n/len(df):>5.1f}%)")

    print(f"\nResults saved to data/evaluation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-domains", nargs="+", default=None)
    parser.add_argument("--seen-only", action="store_true")
    parser.add_argument("--unseen-only", action="store_true")
    args = parser.parse_args()
    evaluate(test_domains=args.test_domains, seen_only=args.seen_only, unseen_only=args.unseen_only)
