import json
import numpy as np
import torch
from pathlib import Path

from negmas.sao import SAONegotiator, ResponseType

from src.agents import make_agent, PORTFOLIO_NAMES
from src.features import extract_all_features, FEATURE_COLUMNS
from src.nn_model import AgentScoreNet, AgentScoreNet2Layer

MODEL_DIR = Path("models")


class MetaAgent(SAONegotiator):
    def __init__(self, *args, model_dir: Path = MODEL_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_dir = model_dir
        self._model = None
        self._scaler_mean = None
        self._scaler_scale = None
        self._meta = None
        self._delegate = None
        self._first_offer_seen = False
        self._load_model()

    def _load_model(self):
        with open(self._model_dir / "meta.json") as f:
            self._meta = json.load(f)

        model_cls = AgentScoreNet2Layer if self._meta["model_class"] == "AgentScoreNet2Layer" else AgentScoreNet
        self._model = model_cls()
        state_dict = torch.load(self._model_dir / "agent_score_net.pt", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        self._scaler_mean = np.load(self._model_dir / "scaler_mean.npy")
        self._scaler_scale = np.load(self._model_dir / "scaler_scale.npy")

    def _select_agent(self, first_offer):
        issues = self.nmi.issues
        features = extract_all_features(issues, self.ufun, first_offer)
        x = np.array([features[c] for c in FEATURE_COLUMNS], dtype=np.float32)
        x = (x - self._scaler_mean) / (self._scaler_scale + 1e-12)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            scores = self._model(x_t).squeeze(0).numpy()

        best_idx = int(np.argmax(scores))
        best_name = PORTFOLIO_NAMES[best_idx]
        return best_name

    def _init_delegate(self, agent_name: str):
        self._delegate = make_agent(agent_name)
        self._delegate._SAONegotiator__end_on_no_response = self._SAONegotiator__end_on_no_response
        self.nmi.add(self._delegate, ufun=self.ufun)

    def propose(self, state):
        if self._delegate is not None:
            return self._delegate.propose(state)
        return self.nmi.random_outcomes(1)[0]

    def respond(self, state, source=None):
        offer = state.current_offer
        if not self._first_offer_seen:
            self._first_offer_seen = True
            agent_name = self._select_agent(offer)
            self._delegate = make_agent(agent_name)
            if hasattr(self._delegate, 'join'):
                self._delegate.join(
                    nmi=self.nmi,
                    state=state,
                    ufun=self.ufun,
                    role=self.role,
                )

        if self._delegate is not None:
            return self._delegate.respond(state, source=source)

        if offer is not None and self.ufun(offer) >= self.ufun.reserved_value:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
