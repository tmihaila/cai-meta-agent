from negmas.sao import SAOMechanism
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun

from src.agents import make_agent
from src.features import extract_all_features


def run_negotiation(issues, ufun_agent: LUFun, ufun_opponent: LUFun,
                    agent_name: str, opponent_name: str, n_steps=100) -> dict:
    session = SAOMechanism(issues=issues, n_steps=n_steps)
    agent = make_agent(agent_name)
    opponent = make_agent(opponent_name)

    session.add(agent, ufun=ufun_agent)
    session.add(opponent, ufun=ufun_opponent)

    result = session.run()

    first_offer = _extract_first_offer(session, agent_idx=0)
    agreement = result.agreement
    agent_util = float(ufun_agent(agreement)) if agreement else float(ufun_agent.reserved_value)
    opp_util = float(ufun_opponent(agreement)) if agreement else float(ufun_opponent.reserved_value)

    features = extract_all_features(issues, ufun_agent, first_offer)

    return {
        **features,
        "agent_type": agent_name,
        "opponent_type": opponent_name,
        "agent_utility": agent_util,
        "opponent_utility": opp_util,
        "nash_product": agent_util * opp_util,
        "agreement_reached": agreement is not None,
        "rounds": result.step,
    }


def _extract_first_offer(session: SAOMechanism, agent_idx: int):
    if not session.history:
        return None
    agent_id = session.negotiator_ids[agent_idx]
    for state in session.history:
        for nid, offer in state.new_offers:
            if nid != agent_id and offer is not None:
                return offer
    return None
