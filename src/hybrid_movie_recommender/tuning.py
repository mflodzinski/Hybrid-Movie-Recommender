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

from .metrics import evaluate_ranking_metrics, evaluate_rmse
from .predictions import save_predictions_to_csv

def hyperparameter_sweep(
    model_class,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    param_grid: dict,
    metric: str = 'rmse'
) -> Tuple[dict, float, dict]:
    print(f"\nHyperparameter Sweep for {model_class.__name__} | Metric: {metric.upper()}")
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    results = {}
    
    ranking_metrics = ['ndcg', 'precision', 'recall', 'f1']
    is_ranking = metric.lower() in ranking_metrics
    
    for param_combo in product(*param_values):
        params = dict(zip(param_names, param_combo))
        print(f"Testing {params}")
        
        model = model_class(**params)
        fit_signature = inspect.signature(model.fit)
        if 'val_data' in fit_signature.parameters:
            model.fit(train_data, val_data)
        else:
            model.fit(train_data)
        
        measure_set_size = val_data.shape[0]
        
        if metric == 'rmse':
            score_val = evaluate_rmse(model, val_data)
            score_train = evaluate_rmse(model, train_data[:measure_set_size])
            print(f"RMSE: val={score_val:.4f}, train={score_train:.4f}")
            results[str(params)] = {'params': params, metric: score_val}
        elif is_ranking:
            ranking_results = evaluate_ranking_metrics(model, val_data, train_data, n=10)
            p, r = ranking_results['precision@10'], ranking_results['recall@10']
            f1 = 2 * (p * r) / (p + r + 1e-10)
            ranking_results['f1@10'] = f1
            
            score_val = ranking_results.get(f'{metric}@10', f1 if metric == 'f1' else 0)
            results[str(params)] = {
                'params': params,
                'ndcg': ranking_results['ndcg@10'],
                'precision': p,
                'recall': r,
                'f1': f1,
                'num_users': ranking_results['num_users_evaluated']
            }
            print(f"Validation {metric.upper()}@10 = {score_val:.4f}")
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    if metric == 'rmse':
        best_key = min(results.keys(), key=lambda k: results[k]['rmse'])
        best_score = results[best_key]['rmse']
    else:
        best_key = max(results.keys(), key=lambda k: results[k][metric])
        best_score = results[best_key][metric]
    
    best_params = results[best_key]['params']
    print(f"\nBest params: {best_params} | Best {metric.upper()}: {best_score:.4f}\n")
    
    return best_params, best_score, results


def train_and_predict_best(
    model_class,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    param_grid: dict,
    output_filepath: str,
    metric: str = 'rmse',
    model_name: Optional[str] = None
) -> Tuple[object, pd.DataFrame]:
    best_params, best_score, _ = hyperparameter_sweep(
        model_class, train_data, val_data, param_grid, metric
    )
    
    if model_name is None:
        model_name = model_class.__name__
    
    print(f"Training final model with best params: {best_params}")
    
    final_model = model_class(**best_params)
    fit_signature = inspect.signature(final_model.fit)
    if 'val_data' in fit_signature.parameters:
        final_model.fit(train_data, val_data)
    else:
        final_model.fit(train_data)
    
    predictions_df = save_predictions_to_csv(
        model=final_model,
        test_data=test_data,
        output_filepath=output_filepath,
        model_name=f"{model_name} {best_params}"
    )
    
    print(f"Done. Best {metric.upper()}: {best_score:.4f} | Saved to: {output_filepath}")
    return final_model, predictions_df
