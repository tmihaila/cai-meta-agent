import sys
import pandas as pd
from pathlib import Path
from itertools import product

from src.domain_loader import list_domains, load_domain
from src.simulation import run_negotiation
from src.agents import PORTFOLIO_NAMES, OPPONENT_NAMES

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
OUTPUT_PATH = Path("data/dataset.csv")
N_STEPS = 100


def generate():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    domain_paths = list_domains(DOMAINS_DIR)
    rows = []
    total = len(domain_paths) * len(PORTFOLIO_NAMES) * len(OPPONENT_NAMES) * 2
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)

        for agent_name, opp_name in product(PORTFOLIO_NAMES, OPPONENT_NAMES):
            for role in ("A", "B"):
                ufun_agent = ufun_a if role == "A" else ufun_b
                ufun_opp = ufun_b if role == "A" else ufun_a

                try:
                    row = run_negotiation(
                        issues, ufun_agent, ufun_opp,
                        agent_name, opp_name, n_steps=N_STEPS,
                    )
                    row["domain_id"] = domain_id
                    row["role"] = role
                    rows.append(row)
                except Exception as e:
                    print(f"ERROR {domain_id} {agent_name} vs {opp_name} role={role}: {e}", file=sys.stderr)

                done += 1
                if done % 100 == 0:
                    print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate()
