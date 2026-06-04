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

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_squared_error

class RatingDataset(Dataset):
    
    def __init__(self, user_ids, item_ids, ratings):
        self.users = torch.LongTensor(user_ids)
        self.items = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorizationSGD:
    def __init__(self, n_factors, learning_rate, n_epochs, use_bias, eval_every, 
                 batch_size=1024, device='auto', l2_reg=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.use_bias = use_bias
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        
        if device == 'auto':
            self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                      else 'cuda' if torch.cuda.is_available() 
                                      else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Device: {self.device}")
        
        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_inv = None
        self.item_inv = None
        
        self.history = {
            'train_rmse': [],
            'val_rmse': [],
            'epochs': []
        }
        self.best_epoch = None
        self.best_val_rmse = float('inf')

    def _initialize_model(self, train_data: pd.DataFrame):
        self.user_mapping = {u: i for i, u in enumerate(train_data['user_id'].unique())}
        self.item_mapping = {i: j for j, i in enumerate(train_data['item_id'].unique())}
        self.user_inv = {i: u for u, i in self.user_mapping.items()}
        self.item_inv = {j: i for i, j in self.item_mapping.items()}

        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        self.P = nn.Embedding(n_users, self.n_factors).to(self.device)
        self.Q = nn.Embedding(n_items, self.n_factors).to(self.device)
        
        nn.init.normal_(self.P.weight, mean=0, std=0.01)
        nn.init.normal_(self.Q.weight, mean=0, std=0.01)

        if self.use_bias:
            self.user_bias = nn.Embedding(n_users, 1).to(self.device)
            self.item_bias = nn.Embedding(n_items, 1).to(self.device)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)
            self.global_mean = float(train_data['rating'].mean())

    def _prepare_training_data(self, train_data: pd.DataFrame):
        user_ids = [self.user_mapping[u] for u in train_data['user_id']]
        item_ids = [self.item_mapping[i] for i in train_data['item_id']]
        ratings = train_data['rating'].values
        
        return RatingDataset(user_ids, item_ids, ratings)

    def _train_one_epoch(self, dataloader, optimizer) -> float:
        self.P.train()
        self.Q.train()
        if self.use_bias:
            self.user_bias.train()
            self.item_bias.train()
        
        total_loss = 0.0
        n_batches = 0
        
        for users, items, ratings in dataloader:
            users = users.to(self.device)
            items = items.to(self.device)
            ratings = ratings.to(self.device)
            
            user_emb = self.P(users)
            item_emb = self.Q(items)
            
            preds = (user_emb * item_emb).sum(dim=1)
            
            if self.use_bias:
                preds = preds + self.global_mean
                preds = preds + self.user_bias(users).squeeze()
                preds = preds + self.item_bias(items).squeeze()
            
            mse_loss = nn.functional.mse_loss(preds, ratings)
            
            l2_loss = 0.0
            if self.l2_reg > 0:
                l2_loss = (
                    self.l2_reg * (user_emb.pow(2).sum() + item_emb.pow(2).sum())
                ) / users.size(0)
                
                if self.use_bias:
                    l2_loss += (
                        self.l2_reg * 0.1 * (
                            self.user_bias(users).pow(2).sum() + 
                            self.item_bias(items).pow(2).sum()
                        )
                    ) / users.size(0)
            
            loss = mse_loss + l2_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += mse_loss.item()
            n_batches += 1
        
        return np.sqrt(total_loss / n_batches)

    def _should_evaluate(self, epoch: int) -> bool:
        return (epoch + 1) % self.eval_every == 0 or (epoch + 1) == self.n_epochs

    def _compute_rmse(self, data: pd.DataFrame) -> float:
        self.P.eval()
        self.Q.eval()
        if self.use_bias:
            self.user_bias.eval()
            self.item_bias.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for _, row in data[['user_id', 'item_id', 'rating']].iterrows():
                pred = self.predict_rating(row['user_id'], row['item_id'])
                preds.append(float(pred))
                trues.append(float(row['rating']))
        
        return float(np.sqrt(mean_squared_error(trues, preds)))

    def _track_metrics(self, epoch: int, train_rmse: float, val_data: pd.DataFrame):
        self.history['train_rmse'].append(train_rmse)
        self.history['epochs'].append(epoch + 1)
        
        if val_data is not None:
            val_rmse = self._compute_rmse(val_data)
            self.history['val_rmse'].append(val_rmse)
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch + 1

    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        self.train_data = train_data
        self._initialize_model(train_data)
        dataset = self._prepare_training_data(train_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        params = list(self.P.parameters()) + list(self.Q.parameters())
        if self.use_bias:
            params += list(self.user_bias.parameters()) + list(self.item_bias.parameters())
        
        optimizer = optim.Adam(params, lr=self.learning_rate)
        
        self.history = {'train_rmse': [], 'val_rmse': [], 'epochs': []}
        self.best_val_rmse = float('inf')
        self.best_epoch = None
        
        for epoch in tqdm(range(self.n_epochs), desc="Training MF"):
            train_rmse = self._train_one_epoch(dataloader, optimizer)
            
            if self._should_evaluate(epoch):
                self._track_metrics(epoch, train_rmse, val_data)
        
        if self.best_epoch is not None:
            print(f"Best val RMSE: {self.best_val_rmse:.4f} at epoch {self.best_epoch}")
        
        return self

    def predict_rating(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return self.global_mean if self.use_bias else 3.0
        
        self.P.eval()
        self.Q.eval()
        if self.use_bias:
            self.user_bias.eval()
            self.item_bias.eval()
        
        with torch.no_grad():
            u = torch.LongTensor([self.user_mapping[user_id]]).to(self.device)
            i = torch.LongTensor([self.item_mapping[item_id]]).to(self.device)
            
            user_emb = self.P(u)
            item_emb = self.Q(i)
            
            pred = torch.dot(user_emb[0], item_emb[0]).item()
            
            if self.use_bias:
                pred += self.global_mean
                pred += self.user_bias(u).item()
                pred += self.item_bias(i).item()
        
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_topk(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        if user_id not in self.user_mapping:
            return []
        
        self.P.eval()
        self.Q.eval()
        if self.use_bias:
            self.user_bias.eval()
            self.item_bias.eval()
        
        with torch.no_grad():
            u = torch.LongTensor([self.user_mapping[user_id]]).to(self.device)
            user_emb = self.P(u)
            
            scores = torch.matmul(user_emb, self.Q.weight.T)[0]
            
            if self.use_bias:
                scores = scores + self.global_mean
                scores = scores + self.user_bias(u).item()
                scores = scores + self.item_bias.weight.squeeze()
            
            scores = scores.cpu().numpy()
        
        seen_items = self.train_data[self.train_data['user_id'] == user_id]['item_id'].values
        seen_idx = [self.item_mapping[i] for i in seen_items if i in self.item_mapping]
        scores[seen_idx] = -np.inf
        
        top_idx = np.argsort(scores)[::-1][:n]
        top_items = [self.item_inv[i] for i in top_idx]
        top_scores = scores[top_idx]
        
        return list(zip(top_items, top_scores))
    
    def get_training_history(self) -> Dict:
        return self.history
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
    
    def plot_training_history(self, figsize=(12, 6), save_path=None):
        if not self.history['epochs']:
            print("No training history available.")
            return
        
        plt.figure(figsize=figsize)
        
        plt.plot(self.history['epochs'], self.history['train_rmse'], 
                label='Train RMSE', marker='o', linewidth=2, markersize=6)
        
        if self.history['val_rmse']:
            plt.plot(self.history['epochs'], self.history['val_rmse'], 
                    label='Validation RMSE', marker='s', linewidth=2, markersize=6)
            
            if self.best_epoch is not None:
                best_idx = self.history['epochs'].index(self.best_epoch)
                best_val_rmse = self.history['val_rmse'][best_idx]
                
                plt.axvline(x=self.best_epoch, color='red', linestyle='--', 
                           label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
                plt.scatter([self.best_epoch], [best_val_rmse], 
                           color='red', s=150, zorder=5, marker='*')
                
                plt.annotate(f'Best: {best_val_rmse:.4f}',
                           xy=(self.best_epoch, best_val_rmse),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Matrix Factorization Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        if self.history['val_rmse']:
            final_val_rmse = self.history['val_rmse'][-1]
            print(f"\nFinal Train: {self.history['train_rmse'][-1]:.4f}, Val: {final_val_rmse:.4f}")
            print(f"Best Val: {self.best_val_rmse:.4f} (Epoch {self.best_epoch})")
            
            if final_val_rmse > self.best_val_rmse:
                diff = final_val_rmse - self.best_val_rmse
                pct = (diff / self.best_val_rmse) * 100
                print(f"Overfitting: +{diff:.4f} ({pct:.2f}%)")
