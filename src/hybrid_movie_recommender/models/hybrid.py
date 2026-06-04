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

class HybridRecommender:
    """Hybrid recommender with learnable weights over pre-trained component models using Ridge Regression (L2)."""

    def __init__(self, models: Dict[str, object], weights: Optional[Dict[str, float]] = None,
                 prediction_files: Optional[Dict[str, str]] = None, alpha: float = 1.0):
        self.models = models
        self.prediction_files = prediction_files
        self.regression_model = None
        self.scaler = StandardScaler()
        self.alpha = alpha
        
        if weights is None:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models.keys()}
        else:
            total = sum(weights.values())
            self.weights = {name: w / total for name, w in weights.items()}
        
        self.train_data = None
        self.global_mean = None
        self.user_means = None
        self.model_names = list(models.keys())
        self.cached_predictions = {}
        
        self._validate_models()

    def _validate_models(self):
        for name, model in self.models.items():
            if not hasattr(model, 'predict_rating'):
                raise ValueError(f"Model '{name}' must have 'predict_rating' method")
            
            if hasattr(model, 'user_mapping') and model.user_mapping is None:
                raise ValueError(f"Model '{name}' appears to be unfitted.")

    def _load_predictions_from_files(self, validation_data: pd.DataFrame):
        """Load predictions from CSV files."""        
        for model_name in self.model_names:
            if model_name not in self.prediction_files:
                raise ValueError(f"No prediction file specified for model '{model_name}'")
            
            filepath = self.prediction_files[model_name]
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Prediction file not found: {filepath}")
            
            pred_df = pd.read_csv(filepath)
            pred_dict = {}
            
            for _, row in pred_df.iterrows():
                key = (int(row['user_id']), int(row['item_id']))
                pred_dict[key] = float(row['predicted_rating'])
            
            self.cached_predictions[model_name] = pred_dict
            print(f"  ✓ Loaded {len(pred_dict)} predictions for {model_name}")

    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None):
        """Learn optimal weights for pre-trained models."""
        self.train_data = train_data
        self.global_mean = train_data['rating'].mean()
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        
        if validation_data is not None:
            if self.prediction_files:
                self._load_predictions_from_files(validation_data)
            self._learn_weights(validation_data)
        
        return self

    def _learn_weights(self, validation_data: pd.DataFrame):
        """Learn weights via Ridge regression (L2 regularization) on validation predictions."""
        X, y = [], []
        valid_pairs, skipped_pairs = 0, 0

        for _, row in tqdm(validation_data.iterrows(), total=len(validation_data), desc="Collecting predictions"):
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            true_rating = float(row['rating'])
            key = (user_id, item_id)
            
            predictions = []
            all_valid = True
            
            for model_name in self.model_names:
                if self.cached_predictions and model_name in self.cached_predictions:
                    pred = self.cached_predictions[model_name].get(key)
                    if pred is None:
                        all_valid = False
                        break
                    predictions.append(pred)
                else:
                    model = self.models[model_name]
                    try:
                        pred = model.predict_rating(user_id, item_id)
                        if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                            pred = self.user_means.get(user_id, self.global_mean)
                        predictions.append(float(pred))
                    except Exception:
                        all_valid = False
                        break

            if all_valid and len(predictions) == len(self.models):
                X.append(predictions)
                y.append(true_rating)
                valid_pairs += 1
            else:
                skipped_pairs += 1

        print(f"\n  Valid pairs: {valid_pairs}")
        print(f"  Skipped pairs: {skipped_pairs}")

        if len(X) == 0:
            print("\n⚠ Warning: No valid predictions. Using equal weights.")
            return

        X = np.array(X)
        y = np.array(y)

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")

        X_scaled = self.scaler.fit_transform(X)
        self.regression_model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.regression_model.fit(X_scaled, y)
        
        learned_weights = self.regression_model.coef_
        intercept = self.regression_model.intercept_

        print(f"\n✓ Ridge Regression model trained (alpha={self.alpha})!")
        print(f"  Intercept (bias): {intercept:.4f}")
        print(f"  Raw coefficients: {learned_weights}")

        abs_weights = np.abs(learned_weights)
        if abs_weights.sum() > 0:
            normalized_weights = abs_weights / abs_weights.sum()
        else:
            normalized_weights = np.ones(len(learned_weights)) / len(learned_weights)

        self.weights = {
            name: float(weight)
            for name, weight in zip(self.model_names, normalized_weights)
        }

        print(f"\n✓ Learned weights (normalized):")
        for name, weight in self.weights.items():
            print(f"    {name:20s}: {weight:.4f} ({weight*100:.1f}%)")

        val_rmse = self._evaluate_on_data(X, y)
        print(f"\n  Validation RMSE: {val_rmse:.4f}")

    def _evaluate_on_data(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate RMSE on given data."""
        if self.regression_model is None:
            return float('inf')
        X_scaled = self.scaler.transform(X)
        predictions = self.regression_model.predict(X_scaled)
        predictions = np.clip(predictions, 1.0, 5.0)
        return float(np.sqrt(np.mean((predictions - y) ** 2)))

    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict a rating using the hybrid model."""
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_rating(user_id, item_id)
                
                if pred is None or (isinstance(pred, float) and np.isnan(pred)):
                    pred = self.user_means.get(user_id, self.global_mean)
                
                predictions.append(float(pred))
            except Exception:
                predictions.append(self.global_mean)

        if len(predictions) == 0:
            return self.global_mean

        if self.regression_model is not None:
            X = np.array([predictions])
            X_scaled = self.scaler.transform(X)
            pred = self.regression_model.predict(X_scaled)[0]
            return float(np.clip(pred, 1.0, 5.0))

        weights = [self.weights[name] for name in self.model_names]
        total_weight = sum(weights)
        if total_weight == 0:
            return float(np.mean(predictions))
        
        weighted_pred = sum(p * w for p, w in zip(predictions, weights)) / total_weight
        return float(np.clip(weighted_pred, 1.0, 5.0))

    def recommend_topk(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a user."""
        if self.train_data is None:
            return []
        
        seen_items = set(self.train_data[self.train_data['user_id'] == user_id]['item_id'].unique())
        all_items = set(self.train_data['item_id'].unique())
        candidate_items = all_items - seen_items
        
        if not candidate_items:
            return []
        
        scores = []
        for item in candidate_items:
            pred = self.predict_rating(user_id, item)
            scores.append((item, float(pred)))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def get_weights(self) -> Dict[str, float]:
        """Return current model weights."""
        return self.weights.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        """Return learned coefficients and normalized weights."""
        if self.regression_model is None:
            return pd.DataFrame({
                'model': self.model_names,
                'weight': [self.weights[name] for name in self.model_names],
            })
        
        return pd.DataFrame({
            'model': self.model_names,
            'weight': [self.weights[name] for name in self.model_names],
        }).sort_values('weight', ascending=False)


class OptimizedHybridRanking:
    def __init__(
        self,
        prediction_files: Dict[str, str],
        train_data: pd.DataFrame,
        relevance_threshold: float = 3.0,
        normalize_per_user: bool = True,
        n_dirichlet: int = 300,
        coord_iters: int = 10,
        coord_grid: Tuple[float, ...] = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        k_eval: int = 10,
        random_state: Optional[int] = 42,
    ):
        self.prediction_files = prediction_files
        self.train_data = train_data
        self.relevance_threshold = relevance_threshold
        self.normalize_per_user = normalize_per_user
        self.n_dirichlet = n_dirichlet
        self.coord_iters = coord_iters
        self.coord_grid = coord_grid
        self.k_eval = int(k_eval)
        self.rng = np.random.default_rng(random_state)
        self.model_names = list(prediction_files.keys())
        self.weights = {name: 1.0 / len(self.model_names) for name in self.model_names}
        self.predictions_df: Optional[pd.DataFrame] = None
        self.model_cols: List[str] = []
        self.user_topk: Dict[int, List[Tuple[int, float]]] = {}
        self._seen_items_by_user = self.train_data.groupby("user_id")["item_id"].apply(set).to_dict()

    @staticmethod
    def _dcg_at_k(rels: np.ndarray, k: int) -> float:
        rels = np.asarray(rels[:k], dtype=float)
        if rels.size == 0:
            return 0.0
        discounts = 1.0 / np.log2(np.arange(2, 2 + rels.size))
        return float(np.sum((2.0 ** rels - 1.0) * discounts))

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = v.size
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.where(u - (cssv - 1) / (np.arange(n) + 1) > 0)[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1.0)
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _fill_missing_scores_per_user_mean(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
        for c in score_cols:
            col_global = df[c].mean()
            df[c] = df.groupby('user_id')[c].transform(lambda s: s.fillna(s.mean())).fillna(col_global)
        return df

    @staticmethod
    def _minmax_per_user(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        def _mm(g):
            g = g.copy()
            for c in cols:
                v = g[c].values
                vmin, vmax = np.nanmin(v), np.nanmax(v)
                g[c] = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            return g
        return df.groupby('user_id', group_keys=False).apply(_mm)

    def _load_and_merge_predictions(self, val_data: pd.DataFrame) -> None:
        pairs = None
        for path in self.prediction_files.values():
            part = pd.read_csv(path, usecols=['user_id', 'item_id']).drop_duplicates()
            pairs = part if pairs is None else pairs.merge(part, on=['user_id', 'item_id'], how='outer')
        df = pairs.copy()
        for model_name, filepath in self.prediction_files.items():
            pred_df = pd.read_csv(filepath, usecols=['user_id', 'item_id', 'predicted_rating'])
            df = df.merge(pred_df.rename(columns={'predicted_rating': f'score__{model_name}'}), on=['user_id', 'item_id'], how='left')
        self.model_cols = [c for c in df.columns if c.startswith('score__')]
        df = self._fill_missing_scores_per_user_mean(df, self.model_cols)
        val_r = val_data[['user_id', 'item_id', 'rating']].copy()
        df = df.merge(val_r, on=['user_id', 'item_id'], how='left')
        df['rel'] = 0.0
        mask = df['rating'].notna()
        df.loc[mask, 'rel'] = (df.loc[mask, 'rating'] >= self.relevance_threshold).astype(float)
        if self.normalize_per_user:
            df = self._minmax_per_user(df, self.model_cols)
        self.predictions_df = df

    def _eval_ndcg(self, w: np.ndarray) -> float:
        df = self.predictions_df.copy()
        df['hybrid'] = df[self.model_cols].values @ w
        ndcgs = []
        k = self.k_eval
        for uid, g in df.groupby('user_id'):
            seen = self._seen_items_by_user.get(uid, set())
            g = g[~g['item_id'].isin(seen)]
            if g.empty:
                continue
            g_sorted = g.sort_values('hybrid', ascending=False)
            top = g_sorted.head(k)
            g_with = g[g['rating'].notna()]
            if g_with.empty:
                continue
            rels = []
            rated_map = dict(zip(g_with['item_id'].values, g_with['rel'].values))
            for iid in top['item_id'].values:
                rels.append(rated_map.get(iid, 0.0))
            dcg = self._dcg_at_k(np.array(rels, dtype=float), k)
            ideal_rels = np.sort(g_with['rel'].values)[::-1][:k]
            idcg = self._dcg_at_k(ideal_rels, k)
            if idcg > 0:
                ndcgs.append(dcg / idcg)
        return 0.0 if not ndcgs else float(np.mean(ndcgs))

    def fit_weights(self, val_data: pd.DataFrame) -> float:
        self._load_and_merge_predictions(val_data)
        m = len(self.model_cols)
        if m == 0:
            raise ValueError("No model score columns found.")
        best_w = np.ones(m) / m
        best_ndcg = self._eval_ndcg(best_w)
        for _ in tqdm(range(self.n_dirichlet), desc="Dirichlet search", leave=False):
            w0 = self.rng.dirichlet(np.ones(m))
            nd = self._eval_ndcg(w0)
            if nd > best_ndcg:
                best_ndcg, best_w = nd, w0
        w = best_w.copy()
        for _ in range(self.coord_iters):
            improved = False
            for j in range(m):
                base = w.copy()
                best_local = best_ndcg
                best_vec = w
                for g in self.coord_grid:
                    w_try = base.copy()
                    w_try[j] = g
                    w_try = self._project_simplex(w_try)
                    nd = self._eval_ndcg(w_try)
                    if nd > best_local:
                        best_local, best_vec = nd, w_try
                if best_local > best_ndcg:
                    best_ndcg, w = best_local, best_vec
                    improved = True
            if not improved:
                break
        self.weights = {c.replace('score__', ''): float(wi) for c, wi in zip(self.model_cols, w)}
        return best_ndcg

    def precompute_topk(self) -> None:
        if self.predictions_df is None:
            raise ValueError("Call fit_weights first.")
        w = np.array([self.weights[c.replace('score__', '')] for c in self.model_cols])
        df = self.predictions_df.copy()
        df['hybrid_score'] = df[self.model_cols].values @ w
        self.user_topk = {}
        k = self.k_eval
        for uid, g in tqdm(df.groupby('user_id'), desc="Precomputing top-k", leave=False):
            seen = self._seen_items_by_user.get(uid, set())
            g = g[~g['item_id'].isin(seen)]
            if g.empty:
                self.user_topk[uid] = []
                continue
            top = g.sort_values('hybrid_score', ascending=False).head(k)
            self.user_topk[uid] = list(zip(top['item_id'].tolist(), top['hybrid_score'].astype(float).tolist()))

    def recommend_topk(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        if n != self.k_eval:
            return self.user_topk.get(user_id, [])[:n]
        return self.user_topk.get(user_id, [])

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if self.predictions_df is None:
            return 0.0
        row = self.predictions_df[(self.predictions_df['user_id'] == user_id) & (self.predictions_df['item_id'] == item_id)]
        if row.empty:
            return 0.0
        w = np.array([self.weights[c.replace('score__', '')] for c in self.model_cols])
        return float(row[self.model_cols].values @ w)

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()
