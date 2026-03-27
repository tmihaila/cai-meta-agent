from negmas.sao import (
    AspirationNegotiator,
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
    NaiveTitForTatNegotiator,
    TimeBasedConcedingNegotiator,
)
from negmas.gb.negotiators.modular import BOANegotiator
from negmas.gb.components import GACTime, GSmithFrequencyModel, GTimeDependentOffering


# BOA = Bidding-Offering-Acceptance modular architecture
def _make_boa(e=0.2, t=0.95):
    return BOANegotiator(
        offering=GTimeDependentOffering(e=e),
        acceptance=GACTime(t=t),
        model=GSmithFrequencyModel(),
    )


# PORTFOLIO: agents used during training dataset generation
PORTFOLIO = {
    "AspirationNegotiator": AspirationNegotiator,
    "BoulwareTBNegotiator": BoulwareTBNegotiator,
    "NaiveTitForTatNegotiator": NaiveTitForTatNegotiator,
    "TimeBasedConcedingNegotiator": TimeBasedConcedingNegotiator,
    "BOA_Boulware": lambda: _make_boa(e=0.2, t=0.95),
}

# OPPONENTS: superset of PORTFOLIO + extra agents used as evaluation opponents
OPPONENTS = {
    **PORTFOLIO,
    "LinearTBNegotiator": LinearTBNegotiator,
    "ConcederTBNegotiator": ConcederTBNegotiator,
    "BOA_Hardliner": lambda: _make_boa(e=0.05, t=0.99),
    "BOA_Conceder": lambda: _make_boa(e=1.0, t=0.90),
    "BOA_Moderate": lambda: _make_boa(e=0.5, t=0.9),
}

PORTFOLIO_NAMES = list(PORTFOLIO.keys())
OPPONENT_NAMES = list(OPPONENTS.keys())


_ALL_AGENTS = {**OPPONENTS}


def register_agents(extra: dict):
    _ALL_AGENTS.update(extra)


def make_agent(name: str):
    factory = _ALL_AGENTS.get(name)
    if factory is None:
        raise KeyError(f"Unknown agent: {name}. Register it first via register_agents().")
    if callable(factory) and isinstance(factory, type):
        return factory()   # class constructor
    return factory()       # lambda factory
