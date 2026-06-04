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

class UserBasedCF:
    def __init__(self, k: int = 50, min_common: int = 2, use_pearson: bool = True):
        self.k = k
        self.min_common = min_common
        self.use_pearson = use_pearson
        self.train_data = None
        self.users = None
        self.items = None
        self.global_mean = None
        self.user_means = None
        self.item_means = None
        self.user_to_index = {}
        self.item_to_index = {}
        self.index_to_user = {}
        self.index_to_item = {}
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.user_neighbors = {}
        self.all_predictions = None

    def _build_matrix(self, df: pd.DataFrame) -> csr_matrix:
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        self.users, self.items = users, items
        self.user_to_index = {u: i for i, u in enumerate(users)}
        self.item_to_index = {it: j for j, it in enumerate(items)}
        self.index_to_user = {i: u for u, i in self.user_to_index.items()}
        self.index_to_item = {j: it for it, j in self.item_to_index.items()}
        rows = df["user_id"].map(self.user_to_index).to_numpy()
        cols = df["item_id"].map(self.item_to_index).to_numpy()
        vals = df["rating"].astype(float).to_numpy()
        return csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))

    def compute_user_similarity_matrix(self, R: csr_matrix) -> np.ndarray:
        X = R.toarray().astype(float)
        if self.use_pearson:
            mask = (X != 0)
            sums = X.sum(axis=1)
            counts = mask.sum(axis=1)
            means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
            X = np.where(mask, X - means[:, np.newaxis], 0.0)
        sims = cosine_similarity(X)
        np.fill_diagonal(sims, 1.0)
        return sims

    def build_topk_neighbors(self, sim_matrix: np.ndarray, k: int) -> Dict[int, List[Tuple[int, float]]]:
        topk = {}
        n_users = sim_matrix.shape[0]
        k_eff = min(k, max(n_users - 1, 0))
        for u in range(n_users):
            if k_eff == 0:
                topk[self.index_to_user[u]] = []
                continue
            sims = sim_matrix[u]
            top_idx = np.argpartition(-sims, range(1, k_eff + 1))[1:k_eff + 1]
            sorted_idx = top_idx[np.argsort(-sims[top_idx])]
            topk[self.index_to_user[u]] = [(self.index_to_user[i], float(sims[i])) for i in sorted_idx if sims[i] > 0]
        return topk

    def fit(self, train_data: pd.DataFrame):
        self.train_data = train_data
        self.global_mean = float(train_data["rating"].mean())
        self.user_means = train_data.groupby("user_id")["rating"].mean().to_dict()
        self.item_means = train_data.groupby("item_id")["rating"].mean().to_dict()
        self.user_item_matrix = self._build_matrix(train_data)
        self.user_similarity_matrix = self.compute_user_similarity_matrix(self.user_item_matrix)
        self.user_neighbors = self.build_topk_neighbors(self.user_similarity_matrix, self.k)
        self._precompute_all_predictions()
        return self

    def _precompute_all_predictions(self):
        R = self.user_item_matrix.toarray().astype(float)
        M = (R != 0).astype(float)
        sims = self.user_similarity_matrix.copy()
        sums = R.sum(axis=1)
        counts = M.sum(axis=1)
        user_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        n_users, n_items = R.shape
        preds = np.zeros((n_users, n_items), dtype=float)

        for u in range(n_users):
            sim_row = sims[u].copy()
            sim_row[u] = 0.0
            pos_mask = (sim_row > 0)
            if not np.any(pos_mask):
                preds[u, :] = user_means[u]
                continue
            pos_idx = np.where(pos_mask)[0]
            k_eff = min(self.k, pos_idx.size)
            top_idx = pos_idx[np.argpartition(-sim_row[pos_idx], k_eff - 1)[:k_eff]]
            w = sim_row[top_idx][:, None]
            R_top = R[top_idx, :]
            M_top = M[top_idx, :]
            mu_top = user_means[top_idx][:, None]
            centered = (R_top - mu_top) * M_top
            num = (w * centered).sum(axis=0)
            den = (np.abs(w) * M_top).sum(axis=0)
            den = np.where(den > 0, den, 1e-8)
            preds[u, :] = user_means[u] + num / den

        self.all_predictions = np.clip(preds, 1.0, 5.0)

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if (user_id not in self.user_to_index) or (item_id not in self.item_to_index):
            return float(self.global_mean)
        u = self.user_to_index[user_id]
        i = self.item_to_index[item_id]
        return float(self.all_predictions[u, i])

    def recommend_topk(self, target_user: int, n: int = 10) -> List[Tuple[int, float]]:
        if target_user not in self.user_to_index:
            return sorted(self.item_means.items(), key=lambda x: x[1], reverse=True)[:n]
        u_idx = self.user_to_index[target_user]
        preds = self.all_predictions[u_idx].copy()
        seen_items = self.train_data.loc[self.train_data["user_id"] == target_user, "item_id"]
        seen_indices = seen_items.map(self.item_to_index).dropna().astype(int).to_numpy()
        preds[seen_indices] = -np.inf
        n_items = min(n, len(preds))
        if n_items == 0:
            return []
        top_idx = np.argpartition(-preds, n_items - 1)[:n_items]
        sorted_idx = top_idx[np.argsort(-preds[top_idx])]
        return [(self.index_to_item[i], float(preds[i])) for i in sorted_idx]


def train_user_based_cf(train_data: pd.DataFrame, k: int = 50, min_common: int = 2) -> UserBasedCF:
    model = UserBasedCF(k=k, min_common=min_common)
    model.fit(train_data)
    return model


class ItemBasedCF:
    def __init__(self, k: int = 50, min_common: int = 2):
        self.k = k
        self.min_common = min_common
        self.train_data = None
        self.item_similarity_matrix = None
        self.users = None
        self.items = None
        self.user_means = None
        self.global_mean = None
        self.item_means = None

        self._user_ids = None
        self._item_ids = None
        self._user_index = None
        self._item_index = None
        self._R = None

    def _build_sparse_matrix(self, train_data: pd.DataFrame) -> csr_matrix:
        self._user_ids = train_data['user_id'].unique()
        self._item_ids = train_data['item_id'].unique()
        self._user_index = {u: i for i, u in enumerate(self._user_ids)}
        self._item_index = {it: j for j, it in enumerate(self._item_ids)}

        rows = train_data['user_id'].map(self._user_index).to_numpy()
        cols = train_data['item_id'].map(self._item_index).to_numpy()
        vals = train_data['rating'].astype(float).fillna(0.0).to_numpy()

        R = csr_matrix((vals, (rows, cols)), shape=(len(self._user_ids), len(self._item_ids)))
        return R

    def compute_item_similarity_matrix(self, train_data: pd.DataFrame) -> pd.DataFrame:
        self._R = self._build_sparse_matrix(train_data)
        sims = cosine_similarity(self._R.T, dense_output=False)
        item_similarity_matrix = pd.DataFrame(
            sims.toarray(), index=self._item_ids, columns=self._item_ids
        )
        np.fill_diagonal(item_similarity_matrix.values, 1.0)
        return item_similarity_matrix

    def get_k_item_neighbors(self, target_item: int, k: int = None) -> List[Tuple[int, float]]:
        if k is None:
            k = self.k
        if self.item_similarity_matrix is None or target_item not in self.item_similarity_matrix.index:
            return []
        
        sims = self.item_similarity_matrix.loc[target_item].drop(labels=target_item, errors='ignore')
        sims = sims[sims > 0]
        
        top_series = sims.sort_values(ascending=False).head(k)
        return list(top_series.items())

    def predict_rating(self, target_user: int, target_item: int) -> float:
        if self.item_similarity_matrix is None or self.train_data is None:
            return float(self.global_mean)

        if target_item not in self.item_similarity_matrix.index:
            return float(self.item_means.get(target_item, self.global_mean))

        user_ratings = self.train_data[self.train_data['user_id'] == target_user]
        if user_ratings.empty:
            return float(self.item_means.get(target_item, self.global_mean))

        user_ratings = user_ratings.set_index('item_id')['rating']

        sims = self.item_similarity_matrix.loc[target_item].drop(labels=target_item, errors='ignore')
        common_items = user_ratings.index.intersection(sims.index)
        
        if len(common_items) == 0:
            return float(self.item_means.get(target_item, self.global_mean))

        sims_common = sims.loc[common_items].dropna()
        
        if len(sims_common) < self.min_common:
            return float(self.item_means.get(target_item, self.global_mean))

        if sims_common.empty:
            return float(self.item_means.get(target_item, self.global_mean))

        top_sims = sims_common.sort_values(ascending=False).head(self.k)
        r_u = user_ratings.reindex(top_sims.index).astype(float)
        s = top_sims.reindex(r_u.index)
        
        mask = r_u.notna() & s.notna()
        if mask.sum() == 0:
            return float(self.item_means.get(target_item, self.global_mean))

        num = (s[mask] * r_u[mask]).sum()
        den = s[mask].abs().sum()

        if den == 0.0 or np.isnan(den):
            return float(self.item_means.get(target_item, self.global_mean))

        result = float(num / den)
        
        if np.isnan(result) or np.isinf(result):
            return float(self.item_means.get(target_item, self.global_mean))

        return float(np.clip(result, 1.0, 5.0))

    def recommend_topk(self, target_user: int, n: int = 10) -> List[Tuple[int, float]]:
        if self.train_data is None or self.item_similarity_matrix is None:
            return []
        
        if target_user not in self.train_data['user_id'].unique():
            popular_items = sorted(self.item_means.items(), key=lambda x: x[1], reverse=True)[:n]
            return popular_items 

        seen_items = set(self.train_data.loc[self.train_data['user_id'] == target_user, 'item_id'].unique())
        candidate_items = self.items - seen_items

        scores = []
        for item in candidate_items:
            pred = self.predict_rating(target_user, item)
            scores.append((item, float(pred)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def fit(self, train_data: pd.DataFrame):
        self.train_data = train_data
        self.users = set(train_data['user_id'].unique())
        self.items = set(train_data['item_id'].unique())
        self.global_mean = float(train_data['rating'].mean())
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()

        self.item_similarity_matrix = self.compute_item_similarity_matrix(train_data)
        return self
