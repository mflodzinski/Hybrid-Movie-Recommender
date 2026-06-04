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

import joblib
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from .models.bpr import BayesianProbabilisticRanking
from .models.content import ContentBasedCF
from .models.hybrid import HybridRecommender, OptimizedHybridRanking
from .models.matrix_factorization import MatrixFactorizationSGD
from .models.neighborhood import ItemBasedCF, UserBasedCF


def _dump(payload: Dict[str, object], filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, filepath)


def save_content_based_model(model: ContentBasedCF, filepath: str):
    """Save ContentBasedCF model."""
    save_dict = {
        'params': {
            'tfidf_max_features': model.tfidf_max_features,
            'svd_dim': model.svd_dim,
            'normalize_emb': model.normalize_emb,
            'text_source': model.text_source,
            'profile_agg': model.profile_agg,
            'positive_threshold': model.positive_threshold
        },
        'global_mean': model.global_mean,
        'item_means': model.item_means,
        'user_means': model.user_means,
        'item_to_idx': model.item_to_idx,
        'idx_to_item': model.idx_to_item,
        'item_embeddings': model.item_embeddings,
        'movies_df': model.movies_df
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved ContentBasedCF to {filepath}")


def save_hybrid_ranking(model: OptimizedHybridRanking, filepath: str):
    """Save OptimizedHybridRanking model."""
    save_dict = {
        'prediction_files': model.prediction_files,
        'relevance_threshold': model.relevance_threshold,
        'normalize_per_user': model.normalize_per_user,
        'n_dirichlet': model.n_dirichlet,
        'coord_iters': model.coord_iters,
        'coord_grid': model.coord_grid,
        'k_eval': model.k_eval,
        'weights': model.weights,
        'model_names': model.model_names,
        'predictions_df': model.predictions_df,
        'model_cols': model.model_cols,
        'user_topk': model.user_topk,
        'train_data': model.train_data
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved OptimizedHybridRanking to {filepath}")


def load_hybrid_ranking(filepath: str) -> OptimizedHybridRanking:
    """Load OptimizedHybridRanking model."""
    save_dict = joblib.load(filepath)
    
    model = OptimizedHybridRanking(
        prediction_files=save_dict['prediction_files'],
        train_data=save_dict['train_data'],
        relevance_threshold=save_dict['relevance_threshold'],
        normalize_per_user=save_dict['normalize_per_user'],
        n_dirichlet=save_dict['n_dirichlet'],
        coord_iters=save_dict['coord_iters'],
        coord_grid=save_dict['coord_grid'],
        k_eval=save_dict['k_eval']
    )
    
    model.weights = save_dict['weights']
    model.model_names = save_dict['model_names']
    model.predictions_df = save_dict['predictions_df']
    model.model_cols = save_dict['model_cols']
    model.user_topk = save_dict['user_topk']
    
    print(f"✓ Loaded OptimizedHybridRanking from {filepath}")
    return model


def load_content_based_model(filepath: str, movies_df: Optional[pd.DataFrame] = None) -> ContentBasedCF:
    """Load ContentBasedCF model."""
    save_dict = joblib.load(filepath)
    
    model = ContentBasedCF(**save_dict['params'])
    model.global_mean = save_dict['global_mean']
    model.item_means = save_dict['item_means']
    model.user_means = save_dict['user_means']
    model.item_to_idx = save_dict['item_to_idx']
    model.idx_to_item = save_dict['idx_to_item']
    model.item_embeddings = save_dict['item_embeddings']
    model.movies_df = save_dict['movies_df']
    model._user_profiles = {}
    
    print(f"✓ Loaded ContentBasedCF from {filepath}")
    return model


def save_user_based_cf(model: UserBasedCF, filepath: str):
    """Save UserBasedCF model."""
    save_dict = {
        'params': {
            'k': model.k,
            'min_common': model.min_common,
            'use_pearson': model.use_pearson
        },
        'global_mean': model.global_mean,
        'user_means': model.user_means,
        'item_means': model.item_means,
        'user_to_index': model.user_to_index,
        'item_to_index': model.item_to_index,
        'index_to_user': model.index_to_user,
        'index_to_item': model.index_to_item,
        'user_similarity_matrix': model.user_similarity_matrix,
        'all_predictions': model.all_predictions,
        'train_data': model.train_data
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved UserBasedCF to {filepath}")


def load_user_based_cf(filepath: str) -> UserBasedCF:
    """Load UserBasedCF model."""
    save_dict = joblib.load(filepath)
    
    model = UserBasedCF(**save_dict['params'])
    model.train_data = save_dict['train_data']
    model.global_mean = save_dict['global_mean']
    model.user_means = save_dict['user_means']
    model.item_means = save_dict['item_means']
    model.user_to_index = save_dict['user_to_index']
    model.item_to_index = save_dict['item_to_index']
    model.index_to_user = save_dict['index_to_user']
    model.index_to_item = save_dict['index_to_item']
    model.user_similarity_matrix = save_dict['user_similarity_matrix']
    model.all_predictions = save_dict['all_predictions']
    model.users = set(model.user_to_index.keys())
    model.items = set(model.item_to_index.keys())
    
    print(f"✓ Loaded UserBasedCF from {filepath}")
    return model


def save_item_based_cf(model: ItemBasedCF, filepath: str):
    """Save ItemBasedCF model."""
    save_dict = {
        'params': {
            'k': model.k,
            'min_common': model.min_common
        },
        'global_mean': model.global_mean,
        'user_means': model.user_means,
        'item_means': model.item_means,
        'item_similarity_matrix': model.item_similarity_matrix,
        'train_data': model.train_data
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved ItemBasedCF to {filepath}")


def load_item_based_cf(filepath: str) -> ItemBasedCF:
    """Load ItemBasedCF model."""
    save_dict = joblib.load(filepath)
    
    model = ItemBasedCF(**save_dict['params'])
    model.train_data = save_dict['train_data']
    model.global_mean = save_dict['global_mean']
    model.user_means = save_dict['user_means']
    model.item_means = save_dict['item_means']
    model.item_similarity_matrix = save_dict['item_similarity_matrix']
    model.users = set(model.train_data['user_id'].unique())
    model.items = set(model.train_data['item_id'].unique())
    
    print(f"✓ Loaded ItemBasedCF from {filepath}")
    return model


def save_matrix_factorization(model: MatrixFactorizationSGD, filepath: str):
    """Save MatrixFactorization model."""
    save_dict = {
        'params': {
            'n_factors': model.n_factors,
            'learning_rate': model.learning_rate,
            'n_epochs': model.n_epochs,
            'use_bias': model.use_bias,
            'eval_every': model.eval_every,
            'batch_size': model.batch_size,
            'l2_reg': model.l2_reg
        },
        'P_weight': model.P.weight.cpu().detach().numpy(),
        'Q_weight': model.Q.weight.cpu().detach().numpy(),
        'user_bias_weight': model.user_bias.weight.cpu().detach().numpy() if model.use_bias else None,
        'item_bias_weight': model.item_bias.weight.cpu().detach().numpy() if model.use_bias else None,
        'global_mean': model.global_mean,
        'user_mapping': model.user_mapping,
        'item_mapping': model.item_mapping,
        'user_inv': model.user_inv,
        'item_inv': model.item_inv,
        'train_data': model.train_data,
        'history': model.history,
        'best_epoch': model.best_epoch,
        'best_val_rmse': model.best_val_rmse
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved MatrixFactorization to {filepath}")


def load_matrix_factorization(filepath: str, device: str = 'auto') -> MatrixFactorizationSGD:
    """Load MatrixFactorization model."""
    save_dict = joblib.load(filepath)
    
    model = MatrixFactorizationSGD(**save_dict['params'], device=device)
    
    n_users = len(save_dict['user_mapping'])
    n_items = len(save_dict['item_mapping'])
    
    model.P = nn.Embedding(n_users, model.n_factors).to(model.device)
    model.Q = nn.Embedding(n_items, model.n_factors).to(model.device)
    
    model.P.weight.data = torch.from_numpy(save_dict['P_weight']).float().to(model.device)
    model.Q.weight.data = torch.from_numpy(save_dict['Q_weight']).float().to(model.device)
    
    if model.use_bias:
        model.user_bias = nn.Embedding(n_users, 1).to(model.device)
        model.item_bias = nn.Embedding(n_items, 1).to(model.device)
        model.user_bias.weight.data = torch.from_numpy(save_dict['user_bias_weight']).float().to(model.device)
        model.item_bias.weight.data = torch.from_numpy(save_dict['item_bias_weight']).float().to(model.device)
    
    model.global_mean = save_dict['global_mean']
    model.user_mapping = save_dict['user_mapping']
    model.item_mapping = save_dict['item_mapping']
    model.user_inv = save_dict['user_inv']
    model.item_inv = save_dict['item_inv']
    model.train_data = save_dict['train_data']
    model.history = save_dict['history']
    model.best_epoch = save_dict['best_epoch']
    model.best_val_rmse = save_dict['best_val_rmse']
    
    print(f"✓ Loaded MatrixFactorization from {filepath}")
    return model


def save_bpr(model: BayesianProbabilisticRanking, filepath: str):
    """Save BPR model."""
    save_dict = {
        'params': {
            'n_factors': model.n_factors,
            'learning_rate': model.learning_rate,
            'n_epochs': model.n_epochs,
            'n_samples': model.n_samples,
            'eval_every': model.eval_every,
            'batch_size': model.batch_size,
            'l2_reg': model.l2_reg
        },
        'P_weight': model.P.weight.cpu().detach().numpy(),
        'Q_weight': model.Q.weight.cpu().detach().numpy(),
        'user_bias_weight': model.user_bias.weight.cpu().detach().numpy(),
        'item_bias_weight': model.item_bias.weight.cpu().detach().numpy(),
        'global_mean': model.global_mean,
        'user_means': model.user_means,
        'user_mapping': model.user_mapping,
        'item_mapping': model.item_mapping,
        'user_inv': model.user_inv,
        'item_inv': model.item_inv,
        'train_data': model.train_data,
        'history': model.history,
        'best_epoch': model.best_epoch,
        'best_val_rmse': model.best_val_rmse
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved BPR to {filepath}")


def load_bpr(filepath: str, device: str = 'auto') -> BayesianProbabilisticRanking:
    """Load BPR model."""
    save_dict = joblib.load(filepath)
    
    model = BayesianProbabilisticRanking(**save_dict['params'], device=device)
    
    n_users = len(save_dict['user_mapping'])
    n_items = len(save_dict['item_mapping'])
    
    model.P = torch.nn.Embedding(n_users, model.n_factors).to(model.device)
    model.Q = torch.nn.Embedding(n_items, model.n_factors).to(model.device)
    model.user_bias = torch.nn.Embedding(n_users, 1).to(model.device)
    model.item_bias = torch.nn.Embedding(n_items, 1).to(model.device)
    
    model.P.weight.data = torch.from_numpy(save_dict['P_weight']).float().to(model.device)
    model.Q.weight.data = torch.from_numpy(save_dict['Q_weight']).float().to(model.device)
    model.user_bias.weight.data = torch.from_numpy(save_dict['user_bias_weight']).float().to(model.device)
    model.item_bias.weight.data = torch.from_numpy(save_dict['item_bias_weight']).float().to(model.device)
    
    model.global_mean = save_dict['global_mean']
    model.user_means = save_dict['user_means']
    model.user_mapping = save_dict['user_mapping']
    model.item_mapping = save_dict['item_mapping']
    model.user_inv = save_dict['user_inv']
    model.item_inv = save_dict['item_inv']
    model.train_data = save_dict['train_data']
    model.history = save_dict['history']
    model.best_epoch = save_dict['best_epoch']
    model.best_val_rmse = save_dict['best_val_rmse']
    
    print(f"✓ Loaded BPR from {filepath}")
    return model


def save_hybrid_model(model: HybridRecommender, filepath: str):
    """Save HybridRecommender model."""
    save_dict = {
        'weights': model.weights,
        'model_names': model.model_names,
        'global_mean': model.global_mean,
        'user_means': model.user_means,
        'train_data': model.train_data,
        'alpha': model.alpha,
        'regression_coef': model.regression_model.coef_ if model.regression_model else None,
        'regression_intercept': model.regression_model.intercept_ if model.regression_model else None,
        'scaler_mean': model.scaler.mean_ if hasattr(model.scaler, 'mean_') else None,
        'scaler_scale': model.scaler.scale_ if hasattr(model.scaler, 'scale_') else None
    }
    _dump(save_dict, filepath)
    print(f"✓ Saved HybridRecommender to {filepath}")


def load_hybrid_model(filepath: str, models: Dict[str, object]) -> HybridRecommender:
    """Load HybridRecommender model."""
    save_dict = joblib.load(filepath)
    
    model = HybridRecommender(models=models, alpha=save_dict['alpha'])
    model.weights = save_dict['weights']
    model.model_names = save_dict['model_names']
    model.global_mean = save_dict['global_mean']
    model.user_means = save_dict['user_means']
    model.train_data = save_dict['train_data']
    
    if save_dict['regression_coef'] is not None:
        model.regression_model = Ridge(alpha=model.alpha)
        model.regression_model.coef_ = save_dict['regression_coef']
        model.regression_model.intercept_ = save_dict['regression_intercept']
        
        if save_dict['scaler_mean'] is not None:
            model.scaler.mean_ = save_dict['scaler_mean']
            model.scaler.scale_ = save_dict['scaler_scale']
    
    print(f"✓ Loaded HybridRecommender from {filepath}")
    return model


MODEL_FILENAMES = {
    'content_rmse': 'content_rmse.pkl',
    'content_ndcg': 'content_ndcg.pkl',
    'userknn_rmse': 'userknn_rmse.pkl',
    'userknn_ndcg': 'userknn_ndcg.pkl',
    'itemknn_rmse': 'itemknn_rmse.pkl',
    'itemknn_ndcg': 'itemknn_ndcg.pkl',
    'mf_rmse': 'mf_rmse.pkl',
    'mf_ndcg': 'mf_ndcg.pkl',
    'bpr_rmse': 'bpr_rmse.pkl',
    'bpr_ndcg': 'bpr_ndcg.pkl',
    'hybrid_rating': 'hybrid_rmse.pkl',
    'hybrid_ranking': 'hybrid_ranking.pkl',
}


def save_all_models(models: Dict[str, object], models_dir: str | Path = 'outputs/models') -> None:
    """Save a dictionary of trained models using the project naming convention."""
    models_dir = Path(models_dir)
    savers = {
        'content_rmse': save_content_based_model,
        'content_ndcg': save_content_based_model,
        'userknn_rmse': save_user_based_cf,
        'userknn_ndcg': save_user_based_cf,
        'itemknn_rmse': save_item_based_cf,
        'itemknn_ndcg': save_item_based_cf,
        'mf_rmse': save_matrix_factorization,
        'mf_ndcg': save_matrix_factorization,
        'bpr_rmse': save_bpr,
        'bpr_ndcg': save_bpr,
        'hybrid_rating': save_hybrid_model,
        'hybrid_ranking': save_hybrid_ranking,
    }
    for key, model in models.items():
        if key not in savers:
            raise KeyError(f"Unknown model key: {key}")
        savers[key](model, models_dir / MODEL_FILENAMES[key])


def load_all_models(models_dir: str | Path = 'outputs/models', device: str = 'auto') -> Dict[str, object]:
    """Load saved project models from disk."""
    models_dir = Path(models_dir)
    loaded = {
        'content_rmse': load_content_based_model(models_dir / MODEL_FILENAMES['content_rmse']),
        'content_ndcg': load_content_based_model(models_dir / MODEL_FILENAMES['content_ndcg']),
        'userknn_rmse': load_user_based_cf(models_dir / MODEL_FILENAMES['userknn_rmse']),
        'userknn_ndcg': load_user_based_cf(models_dir / MODEL_FILENAMES['userknn_ndcg']),
        'itemknn_rmse': load_item_based_cf(models_dir / MODEL_FILENAMES['itemknn_rmse']),
        'itemknn_ndcg': load_item_based_cf(models_dir / MODEL_FILENAMES['itemknn_ndcg']),
        'mf_rmse': load_matrix_factorization(models_dir / MODEL_FILENAMES['mf_rmse'], device=device),
        'mf_ndcg': load_matrix_factorization(models_dir / MODEL_FILENAMES['mf_ndcg'], device=device),
        'bpr_rmse': load_bpr(models_dir / MODEL_FILENAMES['bpr_rmse'], device=device),
        'bpr_ndcg': load_bpr(models_dir / MODEL_FILENAMES['bpr_ndcg'], device=device),
        'hybrid_ranking': load_hybrid_ranking(models_dir / MODEL_FILENAMES['hybrid_ranking']),
    }
    component_models = {
        'Content': loaded['content_rmse'],
        'UserKNN': loaded['userknn_rmse'],
        'ItemKNN': loaded['itemknn_rmse'],
        'MF': loaded['mf_rmse'],
    }
    loaded['hybrid_rating'] = load_hybrid_model(
        models_dir / MODEL_FILENAMES['hybrid_rating'], component_models
    )
    return loaded
