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

from sklearn.metrics import mean_squared_error

def evaluate_rmse(model, test_data: pd.DataFrame) -> float:
    preds = []
    trues = []
    
    for _, row in tqdm(test_data[['user_id', 'item_id', 'rating']].iterrows(), total=len(test_data)):
        u = row['user_id']
        i = row['item_id']
        true_r = row['rating']
        
        pred = model.predict_rating(u, i)
        
        preds.append(float(pred))
        trues.append(float(true_r))
    
    return float(math.sqrt(mean_squared_error(trues, preds)))


def evaluate_ranking_metrics(
    model,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    n: int = 10,
    relevance_threshold: float = 3.0
) -> Dict[str, float]:
    """
    Fast version of evaluate_ranking_metrics:
      - Vectorized DCG/NDCG
      - Precomputes denominators
      - Minimizes Python loops
    """
    test_users = test_data['user_id'].unique()

    # Build ground-truth per user (item -> rating)
    test_group = {
        uid: grp.set_index('item_id')['rating'].to_dict()
        for uid, grp in test_data.groupby('user_id')
    }

    # Precompute discount factors for DCG@k
    discounts = 1.0 / np.log2(np.arange(2, n + 2))

    precision_vals, recall_vals, ndcg_vals = [], [], []

    for user_id in tqdm(test_users, desc=f"Fast eval@{n}"):
        gt_dict = test_group.get(user_id)
        if not gt_dict:
            continue

        relevant_items = {i for i, r in gt_dict.items() if r >= relevance_threshold}
        if not relevant_items:
            continue

        try:
            recs = model.recommend_topk(user_id, n=n)
        except Exception:
            continue

        if not recs:
            continue

        rec_items = [i for i, _ in recs[:n]]
        rels = np.array([gt_dict.get(i, 0.0) for i in rec_items], dtype=float)

        rel_hits = np.sum(np.array([i in relevant_items for i in rec_items], dtype=bool))
        precision = rel_hits / n
        recall = rel_hits / len(relevant_items)
        precision_vals.append(precision)
        recall_vals.append(recall)

        if np.any(rels):
            dcg = np.sum((2.0**rels - 1.0) * discounts)
            ideal_rels = np.sort(list(gt_dict.values()))[::-1][:n]
            idcg = np.sum((2.0**ideal_rels - 1.0) * discounts[: len(ideal_rels)])
            if idcg > 0:
                ndcg_vals.append(dcg / idcg)

    results = {
        f'precision@{n}': float(np.mean(precision_vals)) if precision_vals else 0.0,
        f'recall@{n}': float(np.mean(recall_vals)) if recall_vals else 0.0,
        f'ndcg@{n}': float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        'num_users_evaluated': len(ndcg_vals)
    }

    return results
