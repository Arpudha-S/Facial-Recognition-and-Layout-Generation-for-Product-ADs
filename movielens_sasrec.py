"""
Enhanced SASRec - MovieLens-1M
Clean Strict Version (No Padding)
"""

import hashlib
import random
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# CONSTANTS
# ==========================================================

MIN_SEQ_LEN = 8
MAX_LEN = 50
TITLE_VOCAB_SIZE = 8192
TITLE_MAX_WORDS = 16


# ==========================================================
# UTILS
# ==========================================================

def tokenise_title(title: str) -> List[int]:
    words = str(title).lower().split()[:TITLE_MAX_WORDS]
    return [
        (int(hashlib.md5(w.encode()).hexdigest(), 16) % (TITLE_VOCAB_SIZE - 1)) + 1
        for w in words
    ]


# ==========================================================
# CATALOG
# ==========================================================

class CatalogFeatureBuilder:

    def __init__(self, movies_df, ratings_df):

        self.df = movies_df.reset_index(drop=True)

        rating_means = ratings_df.groupby("MovieID")["Rating"].mean().to_dict()
        self.rating_norm = [
            rating_means.get(mid, 0.0) / 5.0
            for mid in self.df["MovieID"]
        ]

        self.df["genre_list"] = self.df["Genres"].str.split("|")

        all_genres = sorted(
            set(g for genres in self.df["genre_list"] for g in genres)
        )

        self.genre_vocab = {g: i + 1 for i, g in enumerate(all_genres)}

        self.genre_ids = [
            self.genre_vocab.get(genres[0], 0)
            if genres else 0
            for genres in self.df["genre_list"]
        ]

        self.title_ids = []
        for t in self.df["Title"].fillna(""):
            toks = tokenise_title(t)
            padded = toks + [0] * (TITLE_MAX_WORDS - len(toks))
            self.title_ids.append(padded)

        self.pid_to_idx = {
            pid: i for i, pid in enumerate(self.df["MovieID"])
        }

        self.idx_to_pid = {
            i: pid for pid, i in self.pid_to_idx.items()
        }

    @property
    def num_cat1(self):
        return len(self.genre_vocab)

    def get_feature_tensors(self, product_ids, device):

        idxs = [self.pid_to_idx[pid] for pid in product_ids]

        word_ids = torch.tensor(
            [self.title_ids[i] for i in idxs],
            dtype=torch.long,
            device=device,
        )

        genre_ids = torch.tensor(
            [self.genre_ids[i] for i in idxs],
            dtype=torch.long,
            device=device,
        )

        numerics = torch.tensor(
            [[self.rating_norm[i]] for i in idxs],
            dtype=torch.float,
            device=device,
        )

        return word_ids, genre_ids, numerics


# ==========================================================
# PRODUCT ENCODER
# ==========================================================

class ProductEncoder(nn.Module):

    def __init__(self, num_genres, hidden_dim=128):
        super().__init__()

        self.word_emb = nn.Embedding(TITLE_VOCAB_SIZE, 64, padding_idx=0)
        self.genre_emb = nn.Embedding(num_genres + 1, 16, padding_idx=0)
        self.num_proj = nn.Linear(1, 16)

        self.fusion = nn.Sequential(
            nn.Linear(64 + 16 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, word_ids, genre_ids, numerics):

        mask = (word_ids != 0).float()
        emb = self.word_emb(word_ids)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        title = (emb * mask.unsqueeze(-1)).sum(dim=1) / counts

        g = self.genre_emb(genre_ids)
        n = self.num_proj(numerics)

        x = torch.cat([title, g, n], dim=-1)
        return self.fusion(x)

# ==========================================================
# MULTI-INTENT REPRESENTATION
# ==========================================================
class MultiIntentLayer(nn.Module):

    def __init__(self, hidden_dim, num_intents=3):
        super().__init__()

        self.num_intents = num_intents

        self.intent_proj = nn.Linear(hidden_dim, hidden_dim * num_intents)

    def forward(self, session_embedding):

        B, H = session_embedding.shape

        intents = self.intent_proj(session_embedding)

        intents = intents.view(B, self.num_intents, H)

        return intents

# ==========================================================
# TRANSFORMER BLOCK (NO PADDING)
# ==========================================================

class SASRecBlock(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            hidden_dim, 8, batch_first=True, dropout=0.1
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        T = x.size(1)

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=causal_mask,
        )

        x = x + F.dropout(attn_out, p=0.2, training=self.training)
        x = x + self.ffn(self.norm2(x))

        return x


# ==========================================================
# FULL MODEL
# ==========================================================

class EnhancedSASRec(nn.Module):

    def __init__(self, catalog, hidden_dim=256):
        super().__init__()

        self.catalog = catalog
        self.encoder = ProductEncoder(catalog.num_cat1, hidden_dim)
        self.pos_emb = nn.Embedding(MAX_LEN, hidden_dim)
        self.intent_layer = MultiIntentLayer(hidden_dim, num_intents=8)
        self.register_buffer(
            "item_embeddings",
            torch.zeros(len(catalog.idx_to_pid), hidden_dim)
        )

        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_dim) for _ in range(4)]
        )

    def forward(self, pids):

        device = pids.device
        B, T = pids.shape

        flat_ids = pids.reshape(-1).tolist()

        word_ids, genre_ids, numerics = self.catalog.get_feature_tensors(
            flat_ids, device
        )

        prod_emb = self.encoder(word_ids, genre_ids, numerics)
        prod_emb = prod_emb.view(B, T, -1)
        if T > MAX_LEN:
            pids = pids[:, -MAX_LEN:]
            T = MAX_LEN
        pos = torch.arange(T, device=device).unsqueeze(0)
        prod_emb = prod_emb + self.pos_emb(pos)

        for blk in self.blocks:
            prod_emb = blk(prod_emb)

        session = prod_emb[:, -1, :]

        # Multi-Intent
        intents = self.intent_layer(session)

        all_ids = list(self.catalog.idx_to_pid.values())
        word_all, genre_all, num_all = self.catalog.get_feature_tensors(all_ids, device)
        item_emb = self.encoder(word_all, genre_all, num_all)

        scores_list = []

        for i in range(intents.shape[1]):

            s = torch.matmul(intents[:, i, :], item_emb.T)

            scores_list.append(s)

        scores = torch.stack(scores_list).max(dim=0).values

        return scores


    def build_item_embeddings(self, device):

        all_ids = list(self.catalog.idx_to_pid.values())

        word_all, genre_all, num_all = self.catalog.get_feature_tensors(
            all_ids, device
        )

        with torch.no_grad():

            emb = self.encoder(word_all, genre_all, num_all)

        self.item_embeddings.copy_(emb)


# ==========================================================
# TRAINING
# ==========================================================
# ==========================================================
# SEQ2SEQ LOSS (Paper-style future sequence prediction)
# ==========================================================

def seq2seq_loss(model, catalog, batch_inputs, device):

    future_targets = []

    for seq in batch_inputs:

        if len(seq) < 3:
            future_targets.append(seq[-1])
        else:
            future_targets.append(random.choice(seq[1:]))

    target_idx = torch.tensor(
        [catalog.pid_to_idx[t] for t in future_targets],
        dtype=torch.long,
        device=device
    )

    pids = torch.tensor(batch_inputs, dtype=torch.long, device=device)

    scores = model(pids)

    loss = F.cross_entropy(scores, target_idx)

    return loss

def train_model(model, catalog, sequences, epochs=25, batch_size=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = [], []

    for seq in sequences:

        seq = seq[-(MAX_LEN + 1):]

        if len(seq) < MIN_SEQ_LEN + 1:
            continue

        inputs.append(seq[:-1])
        targets.append(seq[-1])

    print(f"[INFO] Training samples: {len(inputs)}")

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        num_batches = 0

        indices = list(range(len(inputs)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):

            batch_idx = indices[i:i + batch_size]

            batch_inputs = [inputs[j] for j in batch_idx]
            batch_targets = [targets[j] for j in batch_idx]

            min_len = min(len(x) for x in batch_inputs)
            batch_inputs = [x[-min_len:] for x in batch_inputs]

            pids = torch.tensor(batch_inputs, dtype=torch.long, device=device)

            target_idx = torch.tensor(
                [catalog.pid_to_idx[t] for t in batch_targets],
                dtype=torch.long,
                device=device,
            )

            scores = model(pids)

            s2i_loss = F.cross_entropy(scores, target_idx)
            s2s_loss = seq2seq_loss(model, catalog, batch_inputs, device)
            loss = s2i_loss + 0.5 * s2s_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/num_batches:.6f}")

    print("[INFO] Training complete.")


# ==========================================================
# MAIN
# ==========================================================

def main():

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

    sequences = (
        ratings.sort_values("Timestamp")
        .groupby("UserID")["MovieID"]
        .apply(list)
        .tolist()
    )

    sequences = [s for s in sequences if len(s) >= MIN_SEQ_LEN + 1]

    catalog = CatalogFeatureBuilder(movies, ratings)
    model = EnhancedSASRec(catalog)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[INFO] Initializing item embeddings...")
    model.build_item_embeddings(device)
    train_model(model, catalog, sequences)

    print("[INFO] Building item embeddings...")
    model.build_item_embeddings(device)

    torch.save(model.state_dict(), "movielens_enhanced_sasrec_full.pth")

if __name__ == "__main__":
    main()