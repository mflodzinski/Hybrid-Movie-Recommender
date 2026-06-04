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

class ItemAverageBaseline:
    """Baseline: Predict rating as average rating of target item."""
    
    def __init__(self):
        self.item_means = None
        self.global_mean = None
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame):
        """Compute item average ratings."""
        self.train_data = train_data
        self.global_mean = float(train_data['rating'].mean())
        self.item_means = train_data.groupby('item_id')['rating'].mean().to_dict()
        return self
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating as item average."""
        rating = self.item_means.get(item_id, self.global_mean)
        return float(np.clip(rating, 1.0, 5.0))


class MeanHybridRating:
    """Baseline: Average predictions from all component models."""
    
    def __init__(self, models: Dict[str, object]):
        self.models = models
        self.model_names = list(models.keys())
        self.train_data = None
        self.global_mean = None
        self.user_means = None
    
    def fit(self, train_data: pd.DataFrame):
        """Store training data statistics."""
        self.train_data = train_data
        self.global_mean = float(train_data['rating'].mean())
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        return self
    
    def predict_rating(self, user_id: int, item_id: int) -> float:
        """Predict rating as average of all component predictions."""
        predictions = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict_rating(user_id, item_id)
                if pred is not None and not np.isnan(pred):
                    predictions.append(float(pred))
            except Exception:
                pass
        
        if len(predictions) == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        avg_pred = float(np.mean(predictions))
        return float(np.clip(avg_pred, 1.0, 5.0))


class RandomRecommender:
    """Baseline: Recommend random items."""
    
    def __init__(self, seed: int = 10):
        self.seed = seed
        self.train_data = None
        random.seed(seed)
    
    def fit(self, train_data: pd.DataFrame):
        self.train_data = train_data
        return self

    def recommend_topk(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommend random items."""
        seen_items = set(self.train_data[self.train_data['user_id'] == user_id]['item_id'].unique())
        all_items = set(self.train_data['item_id'].unique())
        candidate_items = list(all_items - seen_items)
        
        if not candidate_items: return []
        n_select = min(n, len(candidate_items))
        selected_items = random.sample(candidate_items, n_select)
        scores = [(item, random.random()) for item in selected_items]
        return scores


class PopularityRecommender:
    """Baseline: Recommend most popular items (by number of ratings)."""
    def __init__(self):
        self.item_popularity = None
        self.train_data = None
    
    def fit(self, train_data: pd.DataFrame):
        """Compute item popularity (rating count) and average ratings."""
        self.train_data = train_data
        self.item_popularity = train_data.groupby('item_id').size().to_dict()
        return self

    
    def recommend_topk(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """Recommend most popular items."""
        seen_items = set(self.train_data[self.train_data['user_id'] == user_id]['item_id'].unique())
        all_items = set(self.train_data['item_id'].unique())
        candidate_items = all_items - seen_items
        
        scores = [(item, float(self.item_popularity.get(item, 0))) 
                  for item in candidate_items]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
