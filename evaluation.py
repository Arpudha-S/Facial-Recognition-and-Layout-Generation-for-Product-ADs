"""
Research-style evaluation for Sequential Recommendation

Metrics:
HR@K
NDCG@K
MRR

Uses:
1 positive item
99 negative items
"""

import torch
import numpy as np
import random
import pandas as pd

MAX_LEN = 50
MIN_SEQ_LEN = 8
MODEL_PATH ="movielens_enhanced_sasrec_full.pth"
from movielens_sasrec import CatalogFeatureBuilder, EnhancedSASRec

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def hit_rate(ranklist, target):
    return 1 if target in ranklist else 0


def ndcg(ranklist, target):
    if target in ranklist:
        idx = ranklist.index(target)
        return 1 / np.log2(idx + 2)
    return 0


def mrr(ranklist, target):
    if target in ranklist:
        idx = ranklist.index(target)
        return 1 / (idx + 1)
    return 0

# ---------------------------------------------------
# Negative Sampling
# ---------------------------------------------------

def sample_negatives(catalog, target, history, n=99):

    all_items = list(catalog.pid_to_idx.keys())
    history = set(history)
    negatives = set()

    while len(negatives) < n:
        item = random.choice(all_items)

        if item != target and item not in history:
            negatives.add(item)

    return list(negatives)

# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate_model(MODEL, catalog, sequences, k=10, min_seq_len=8):

    device = next(MODEL.parameters()).device

    HR, NDCG, MRR = [], [], []

    MODEL.eval()

    with torch.no_grad():

        for seq in sequences:
            if len(seq) < MIN_SEQ_LEN + 1:
                continue

            target = seq[-1]
            input_seq = seq[:-1]
  
            input_seq = input_seq[-MAX_LEN:]

            pids = torch.tensor(
                [input_seq],
                dtype=torch.long,
                device=device
            )

            scores = MODEL(pids)[0]

            # -------------------------------
            # Candidate generation
            # -------------------------------

            negatives = sample_negatives(
                catalog,
                target,
                input_seq,
                99
            )

            candidates = negatives + [target]

            candidate_idx = []
            valid_candidates = []

            for item in candidates:
                if item in catalog.pid_to_idx:
                    candidate_idx.append(catalog.pid_to_idx[item])
                    valid_candidates.append(item)

            if len(candidate_idx) == 0:
                continue

            candidate_scores = scores[candidate_idx]

            ranking = torch.argsort(
                candidate_scores,
                descending=True
            )
            ranking = ranking.cpu().numpy()
            ranked_items = [
                valid_candidates[i]
                for i in ranking[:k]
            ]

            HR.append(hit_rate(ranked_items, target))
            NDCG.append(ndcg(ranked_items, target))
            MRR.append(mrr(ranked_items, target))

    if len(HR) == 0:
        return {
            "HR@10": 0,
            "NDCG@10": 0,
            "MRR": 0
        }
    
    results = {
        f"HR@{k}": float(np.mean(HR)),
        f"NDCG@{k}": float(np.mean(NDCG)),
        "MRR": float(np.mean(MRR)),
    }

    return results

def main():
    print("\nStarting evaluation...")
    movies = pd.read_csv(
        "movies.dat",
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )

    ratings = pd.read_csv(
        "ratings.dat",
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
    )

    ratings["MovieID"] = ratings["MovieID"].astype(int)
    movies["MovieID"] = movies["MovieID"].astype(int)

    sequences = (
        ratings.sort_values("Timestamp")
        .groupby("UserID")["MovieID"]
        .apply(list)
        .tolist()
    )

    sequences = [s for s in sequences if len(s) >= MIN_SEQ_LEN + 1]
    print("Total sequences:", len(sequences))

    catalog = CatalogFeatureBuilder(movies, ratings)
    model = EnhancedSASRec(catalog)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    
    metrics = evaluate_model(
        model,
        catalog,
        sequences,
        k=10
    )

    print("\nEvaluation Results")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()