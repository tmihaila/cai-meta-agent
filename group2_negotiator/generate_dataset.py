import sys
import pandas as pd
from pathlib import Path
from itertools import product

from negmas.sao import SAOMechanism
from negmas.gb.negotiators.modular import BOANegotiator
from negmas.gb.components import GACTime, GTimeDependentOffering

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
from src.features import extract_all_features, FEATURE_COLUMNS
from src.agents import make_agent, OPPONENT_NAMES, register_agents

UNSEEN_OPPONENTS = {
    "MiCRONegotiator": MiCRONegotiator,
    "NiceNegotiator": NiceNegotiator,
    "RandomNegotiator": RandomNegotiator,
    "ToughNegotiator": ToughNegotiator,
    "SimpleTitForTatNegotiator": SimpleTitForTatNegotiator,
    "TopFractionNegotiator": TopFractionNegotiator,
    "FirstOfferOrientedTBNegotiator": FirstOfferOrientedTBNegotiator,
}
register_agents(UNSEEN_OPPONENTS)

ALL_OPPONENTS = OPPONENT_NAMES + list(UNSEEN_OPPONENTS.keys())

DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains_python"
OUTPUT_PATH = Path("data/dataset.csv")
N_STEPS = 100
E_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ALPHA = 0.7


def make_td_agent(e):
    return BOANegotiator(
        offering=GTimeDependentOffering(e=e),
        acceptance=GACTime(t=0.95),
    )


def extract_first_offer(session, agent_idx=0):
    if not session.history:
        return None
    agent_id = session.negotiator_ids[agent_idx]
    for state in session.history:
        for nid, offer in state.new_offers:
            if nid != agent_id and offer is not None:
                return offer
    return None


def run_one(issues, ufun_agent, ufun_opp, e, opp_name, n_steps):
    session = SAOMechanism(issues=issues, n_steps=n_steps)
    agent = make_td_agent(e)
    opponent = make_agent(opp_name)

    session.add(agent, ufun=ufun_agent)
    session.add(opponent, ufun=ufun_opp)

    result = session.run()
    first_offer = extract_first_offer(session)
    agreement = result.agreement
    au = float(ufun_agent(agreement)) if agreement else float(ufun_agent.reserved_value)
    ou = float(ufun_opp(agreement)) if agreement else float(ufun_opp.reserved_value)

    features = extract_all_features(issues, ufun_agent, first_offer)
    return {
        **features,
        "e": e,
        "opponent_type": opp_name,
        "agent_utility": au,
        "opponent_utility": ou,
        "nash_product": au * ou,
        "score": ALPHA * au + (1 - ALPHA) * au * ou,
        "agreement_reached": agreement is not None,
        "rounds": result.step,
    }


def generate():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    domain_paths = list_domains(DOMAINS_DIR)
    rows = []
    total = len(domain_paths) * 2 * len(ALL_OPPONENTS) * len(E_VALUES)
    done = 0

    for domain_path in domain_paths:
        domain_id = domain_path.stem
        issues, ufun_a, ufun_b = load_domain(domain_path)

        for role in ("A", "B"):
            ufun_agent = ufun_a if role == "A" else ufun_b
            ufun_opp = ufun_b if role == "A" else ufun_a

            for opp_name in ALL_OPPONENTS:
                for e in E_VALUES:
                    try:
                        row = run_one(issues, ufun_agent, ufun_opp, e, opp_name, N_STEPS)
                        row["domain_id"] = domain_id
                        row["role"] = role
                        rows.append(row)
                    except Exception as ex:
                        print(f"ERROR {domain_id} e={e} vs {opp_name} role={role}: {ex}", file=sys.stderr)

                    done += 1
                    if done % 500 == 0:
                        print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    generate()
