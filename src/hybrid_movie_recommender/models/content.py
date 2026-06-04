from __future__ import annotations

import inspect
import math
import os
import pickle
import random
import re
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

class ContentBasedCF:
    
    def __init__(self,
                 tfidf_max_features: int = 20000,
                 svd_dim: int = None,
                 normalize_emb: bool = True,
                 text_source: str = "full_text",
                 profile_agg: str = "weighted_avg",
                 positive_threshold: float = 4.0):
        self.tfidf_max_features = tfidf_max_features
        self.svd_dim = svd_dim
        self.normalize_emb = normalize_emb
        self.text_source = text_source
        self.profile_agg = profile_agg
        self.positive_threshold = positive_threshold

        self.train_data = None
        self.items = None
        self.users = None

        self.global_mean = None
        self.item_means = None
        self.user_means = None

        self.movies_df = None
        self.item_to_idx = None
        self.idx_to_item = None
        self.item_embeddings = None

        self._user_profiles = {}

    def _prepare_movie_features(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        df = movies_df.copy()
        df['title'] = df['title'].fillna('')
        df['genres'] = df['genres'].fillna('')
        df['description'] = df['description'].fillna('')
        df['titlegenres'] = (df['title'] + ' ' + df['genres']).str.lower()
        df['full_text'] = (df['titlegenres'] + ' ' + df['description']).str.lower()
        return df

    def _tokenizer(self, s: str) -> List[str]:
        return re.findall(r"\w+|\S", s)

    def _compute_item_embeddings(self, corpus: List[str]) -> np.ndarray:
        tfidf = TfidfVectorizer(
            tokenizer=self._tokenizer,
            lowercase=False,
            ngram_range=(1, 2),
            stop_words='english',
            max_features=self.tfidf_max_features
        )
        X = tfidf.fit_transform(corpus)
        if self.svd_dim is not None and 0 < self.svd_dim < X.shape[1]:
            svd = TruncatedSVD(n_components=self.svd_dim, random_state=42)
            X = svd.fit_transform(X)
        else:
            X = X.toarray()
        if self.normalize_emb:
            X = normalize(X)
        return X

    def fit(self, train_data: pd.DataFrame, movies_df: pd.DataFrame):
        self.train_data = train_data
        self.users = set(train_data['user_id'].unique())
        self.items = set(train_data['item_id'].unique())
        self.global_mean = float(train_data['rating'].mean())
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()

        self.movies_df = self._prepare_movie_features(movies_df)

        if self.text_source not in {'description', 'titlegenres', 'full_text'}:
            raise ValueError(f"Unknown text_source={self.text_source}")
        text_col = self.text_source

        self.item_to_idx = {iid: idx for idx, iid in enumerate(self.movies_df['item_id'])}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}

        self.item_embeddings = self._compute_item_embeddings(self.movies_df[text_col].tolist())
        self._user_profiles.clear()
        return self

    def _build_user_profile(self, user_id: int):
        rows = self.train_data[self.train_data['user_id'] == user_id]
        if rows.empty:
            return None

        if self.profile_agg == 'avg_pos':
            rows = rows[rows['rating'] >= self.positive_threshold]
            if rows.empty:
                return None

        rows = rows[rows['item_id'].isin(self.item_to_idx.keys())]
        if rows.empty:
            return None

        idxs = [self.item_to_idx[i] for i in rows['item_id']]
        embs = self.item_embeddings[idxs]

        if self.profile_agg == 'weighted_avg':
            weights = rows['rating'].to_numpy(dtype=float)
            prof = np.average(embs, axis=0, weights=weights)
        else:
            prof = embs.mean(axis=0)

        prof = prof / (np.linalg.norm(prof) + 1e-9)
        return prof

    def _get_user_profile(self, user_id: int):
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]
        prof = self._build_user_profile(user_id)
        if prof is not None:
            self._user_profiles[user_id] = prof
        return prof

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if self.train_data is None or self.item_embeddings is None:
            return float(np.clip(self.global_mean if self.global_mean is not None else 3.0, 1.0, 5.0))

        user_known = user_id in self.user_means
        item_known = item_id in self.item_to_idx

        if not user_known and not item_known:
            return float(np.clip(self.global_mean, 1.0, 5.0))
        if not user_known and item_known:
            return float(np.clip(self.item_means.get(item_id, self.global_mean), 1.0, 5.0))
        if user_known and not item_known:
            return float(np.clip(self.user_means.get(user_id, self.global_mean), 1.0, 5.0))

        profile = self._get_user_profile(user_id)
        if profile is None:
            um = self.user_means.get(user_id, self.global_mean)
            im = self.item_means.get(item_id, np.nan)
            return float(np.clip(um if np.isnan(im) else (um + im) / 2, 1.0, 5.0))

        item_vec = self.item_embeddings[self.item_to_idx[item_id]]
        item_vec = item_vec / (np.linalg.norm(item_vec) + 1e-9)

        sim = float(np.dot(profile, item_vec))
        um = self.user_means.get(user_id, self.global_mean)
        pred = um + 2.0 * sim
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_topk(self, target_user: int, n: int = 10) -> List[Tuple[int, float]]:
        if self.train_data is None or self.item_embeddings is None:
            return []
        if target_user not in self.users:
            avail = [(iid, self.item_means.get(iid, self.global_mean))
                     for iid in self.item_to_idx.keys()]
            return sorted(avail, key=lambda x: x[1], reverse=True)[:n]

        seen = set(self.train_data[self.train_data['user_id'] == target_user]['item_id'])
        candidates = [iid for iid in self.item_to_idx.keys() if iid not in seen]
        if not candidates:
            return []

        prof = self._get_user_profile(target_user)
        if prof is None:
            avail = [(iid, self.item_means.get(iid, self.global_mean)) for iid in candidates]
            return sorted(avail, key=lambda x: x[1], reverse=True)[:n]

        idxs = [self.item_to_idx[iid] for iid in candidates]
        cand = self.item_embeddings[idxs]
        cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
        sims = cand @ prof

        base = self.user_means.get(target_user, self.global_mean)
        scores = np.clip(base + 2.0 * sims, 1.0, 5.0)
        top = np.argsort(scores)[::-1][:n]
        return [(candidates[i], float(scores[i])) for i in top]


class ContentBasedCFWithMovies(ContentBasedCF):
    def __init__(self, movies_df, **kwargs):
        super().__init__(**kwargs)
        self._movies_df_external = movies_df
    
    def fit(self, train_data: pd.DataFrame):
        return super().fit(train_data, self._movies_df_external)
