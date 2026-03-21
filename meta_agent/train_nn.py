import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.agents import PORTFOLIO_NAMES
from src.features import FEATURE_COLUMNS
from src.nn_model import AgentScoreNet, AgentScoreNet2Layer

ALPHA = 0.7
DATA_PATH = Path("data/dataset.csv")
MODEL_DIR = Path("models")


def compute_score(df: pd.DataFrame) -> pd.Series:
    return ALPHA * df["agent_utility"] + (1 - ALPHA) * df["nash_product"]


def pivot_to_agent_scores(df: pd.DataFrame):
    df = df.copy()
    df["score"] = compute_score(df)
    group_cols = ["domain_id", "role", "opponent_type"]
    feature_rows = []
    score_rows = []

    for keys, group in df.groupby(group_cols):
        for agent_name in PORTFOLIO_NAMES:
            agent_rows = group[group["agent_type"] == agent_name]
            if agent_rows.empty:
                continue
            row = agent_rows.iloc[0]
            feature_rows.append({
                "domain_id": keys[0],
                "role": keys[1],
                "opponent_type": keys[2],
                **{col: row[col] for col in FEATURE_COLUMNS},
            })
            scores = {}
            for an in PORTFOLIO_NAMES:
                ar = group[group["agent_type"] == an]
                scores[an] = float(ar["score"].values[0]) if not ar.empty else 0.0
            score_rows.append(scores)
            break

    feat_df = pd.DataFrame(feature_rows)
    score_df = pd.DataFrame(score_rows)
    return feat_df, score_df


def train(model_class=AgentScoreNet, epochs=300, lr=1e-3, patience=20, batch_size=64):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    feat_df, score_df = pivot_to_agent_scores(df)
    domains = feat_df["domain_id"].values

    X = feat_df[FEATURE_COLUMNS].values.astype(np.float32)
    Y = score_df[PORTFOLIO_NAMES].values.astype(np.float32)

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, groups=domains))

    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_t = torch.tensor(X_train)
    Y_train_t = torch.tensor(Y_train)
    X_val_t = torch.tensor(X_val)
    Y_val_t = torch.tensor(Y_val)

    model = model_class()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X_train_t[idx])
            loss = loss_fn(pred, Y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t).item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: train={epoch_loss/n_batches:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "agent_score_net.pt")
    np.save(MODEL_DIR / "scaler_mean.npy", scaler.mean_)
    np.save(MODEL_DIR / "scaler_scale.npy", scaler.scale_)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "portfolio": PORTFOLIO_NAMES,
        "alpha": ALPHA,
        "best_val_loss": best_val_loss,
        "model_class": model_class.__name__,
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved. Best val loss: {best_val_loss:.4f}")
    return model, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=1, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    cls = AgentScoreNet if args.layers == 1 else AgentScoreNet2Layer
    train(model_class=cls, epochs=args.epochs, lr=args.lr, patience=args.patience)
