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
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_squared_error

class BPRDataset(Dataset):
    
    def __init__(self, triples):
        self.triples = torch.LongTensor(triples)
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]


class BayesianProbabilisticRanking:
    """Bayesian Personalized Ranking with L2 regularization."""

    def __init__(self, n_factors, learning_rate, n_epochs, n_samples, eval_every, 
                 batch_size, device, l2_reg=0.01):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_samples = n_samples
        self.eval_every = eval_every
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        
        if device == 'auto':
            self.device = torch.device('mps' if torch.backends.mps.is_available() 
                                      else 'cuda' if torch.cuda.is_available() 
                                      else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"L2 Regularization: {self.l2_reg}")
        
        self.P = None
        self.Q = None
        self.user_bias = None
        self.item_bias = None
        self.user_mapping = None
        self.item_mapping = None
        self.user_inv = None
        self.item_inv = None
        self.global_mean = None
        self.user_means = None
        
        self.history = {
            'train_loss': [],
            'val_rmse': [],
            'epochs': []
        }
        self.best_epoch = None
        self.best_val_rmse = float('inf')

    def _generate_training_triples(self, ratings):
        triples = []
        user_items = ratings.groupby('user_id')['item_id'].apply(set).to_dict()
        all_items = set(ratings['item_id'].unique())

        for user_id, rated_items in user_items.items():
            user_idx = self.user_mapping[user_id]
            unrated_items = list(all_items - rated_items)
            if len(unrated_items) == 0:
                continue
            
            rated_items_list = list(rated_items)
            for pos_item_id in rated_items_list:
                pos_item_idx = self.item_mapping[pos_item_id]
                n_neg = min(self.n_samples, len(unrated_items))
                neg_item_ids = np.random.choice(unrated_items, size=n_neg, replace=False)
                for neg_item_id in neg_item_ids:
                    neg_item_idx = self.item_mapping[neg_item_id]
                    triples.append([user_idx, pos_item_idx, neg_item_idx])
        
        return np.array(triples)

    def _should_evaluate(self, epoch):
        return (epoch + 1) % self.eval_every == 0 or (epoch + 1) == self.n_epochs or epoch == 0

    def _compute_rmse_gpu(self, data):
        self.P.eval()
        self.Q.eval()
        if self.user_bias is not None:
            self.user_bias.eval()
            self.item_bias.eval()
        
        user_ids = []
        item_ids = []
        true_ratings = []
        
        for _, row in data[['user_id', 'item_id', 'rating']].iterrows():
            user_ids.append(row['user_id'])
            item_ids.append(row['item_id'])
            true_ratings.append(row['rating'])
        
        preds = []
        batch_size = 1000
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]
            batch_items = item_ids[i:i+batch_size]
            
            batch_preds = []
            for u, it in zip(batch_users, batch_items):
                pred = self.predict_rating(u, it)
                batch_preds.append(pred)
            
            preds.extend(batch_preds)
        
        preds = np.array(preds)
        true_ratings = np.array(true_ratings)
        
        return float(np.sqrt(mean_squared_error(true_ratings, preds)))

    def _track_metrics(self, epoch, avg_loss, val_data):
        self.history['train_loss'].append(avg_loss)
        self.history['epochs'].append(epoch + 1)
        
        if val_data is not None:
            val_rmse = self._compute_rmse_gpu(val_data)
            self.history['val_rmse'].append(val_rmse)
            
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch + 1

    def fit(self, train_data, val_data):
        self.train_data = train_data
        self.user_mapping = {u: i for i, u in enumerate(train_data['user_id'].unique())}
        self.item_mapping = {i: j for j, i in enumerate(train_data['item_id'].unique())}
        self.user_inv = {i: u for u, i in self.user_mapping.items()}
        self.item_inv = {j: i for i, j in self.item_mapping.items()}

        self.global_mean = float(train_data['rating'].mean())
        self.user_means = train_data.groupby('user_id')['rating'].mean().to_dict()
        print(f"Global mean: {self.global_mean:.4f}")
        
        n_users = len(self.user_mapping)
        n_items = len(self.item_mapping)

        self.P = torch.nn.Embedding(n_users, self.n_factors).to(self.device)
        self.Q = torch.nn.Embedding(n_items, self.n_factors).to(self.device)
        
        torch.nn.init.normal_(self.P.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.Q.weight, mean=0, std=0.01)

        self.user_bias = torch.nn.Embedding(n_users, 1).to(self.device)
        self.item_bias = torch.nn.Embedding(n_items, 1).to(self.device)
        torch.nn.init.zeros_(self.user_bias.weight)
        torch.nn.init.zeros_(self.item_bias.weight)

        optimizer = optim.Adam(
            list(self.P.parameters()) + 
            list(self.Q.parameters()) +
            list(self.user_bias.parameters()) +
            list(self.item_bias.parameters()),
            lr=self.learning_rate
        )

        self.history = {'train_loss': [], 'val_rmse': [], 'epochs': []}
        self.best_val_rmse = float('inf')
        self.best_epoch = None

        for epoch in tqdm(range(self.n_epochs), desc="Training BPR"):
            self.P.train()
            self.Q.train()
            self.user_bias.train()
            self.item_bias.train()
            
            triples = self._generate_training_triples(train_data)
            np.random.shuffle(triples)
            
            dataset = BPRDataset(triples)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                users = batch[:, 0]
                pos_items = batch[:, 1]
                neg_items = batch[:, 2]
                
                user_emb = self.P(users)
                pos_item_emb = self.Q(pos_items)
                neg_item_emb = self.Q(neg_items)
                
                pos_scores = (user_emb * pos_item_emb).sum(dim=1)
                neg_scores = (user_emb * neg_item_emb).sum(dim=1)
                
                bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
                
                l2_loss = 0.0
                if self.l2_reg > 0:
                    l2_loss = (
                        self.l2_reg * (
                            user_emb.pow(2).sum() + 
                            pos_item_emb.pow(2).sum() + 
                            neg_item_emb.pow(2).sum()
                        )
                    ) / users.size(0)
                    
                    l2_loss += (
                        self.l2_reg * 0.1 * (
                            self.user_bias(users).pow(2).sum() +
                            self.item_bias(pos_items).pow(2).sum() +
                            self.item_bias(neg_items).pow(2).sum()
                        )
                    ) / users.size(0)
                
                loss = bpr_loss + l2_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += bpr_loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            
            if self._should_evaluate(epoch):
                self._track_metrics(epoch, avg_loss, val_data)
                if self.history['val_rmse']:
                    print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {avg_loss:.4f} - Val RMSE: {self.history['val_rmse'][-1]:.4f}")
        
        if self.best_epoch is not None:
            print(f"\nBest validation RMSE: {self.best_val_rmse:.4f} at epoch {self.best_epoch}")
        
        return self

    def predict_rating(self, user_id, item_id):
        """Predict rating using bias terms for better accuracy."""
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            if user_id in self.user_means:
                return float(self.user_means[user_id])
            return float(self.global_mean)
        
        self.P.eval()
        self.Q.eval()
        self.user_bias.eval()
        self.item_bias.eval()
        
        with torch.no_grad():
            u = torch.LongTensor([self.user_mapping[user_id]]).to(self.device)
            i = torch.LongTensor([self.item_mapping[item_id]]).to(self.device)
            
            user_emb = self.P(u)
            item_emb = self.Q(i)
            
            score = (
                self.global_mean +
                self.user_bias(u).item() +
                self.item_bias(i).item() +
                torch.dot(user_emb[0], item_emb[0]).item()
            )
            
            return float(np.clip(score, 1.0, 5.0))

    def recommend_topk(self, user_id, n=10):
        """Generate top-N recommendations for ranking tasks."""
        if user_id not in self.user_mapping:
            return []
        
        self.P.eval()
        self.Q.eval()
        self.user_bias.eval()
        self.item_bias.eval()
        
        with torch.no_grad():
            u = torch.LongTensor([self.user_mapping[user_id]]).to(self.device)
            user_emb = self.P(u)
            
            scores = torch.matmul(user_emb, self.Q.weight.T)[0]
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

    def get_training_history(self):
        return self.history
    
    def get_best_epoch(self):
        return self.best_epoch
    
    def plot_training_history(self, figsize=(12, 6), save_path=None):
        if not self.history['epochs']:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.plot(self.history['epochs'], self.history['train_loss'], 
                label='Train Loss', marker='o', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('BPR Training Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        if self.history['val_rmse']:
            ax2.plot(self.history['epochs'], self.history['val_rmse'], 
                    label='Validation RMSE', marker='s', linewidth=2, markersize=6, color='green')
            
            if self.best_epoch is not None:
                best_idx = self.history['epochs'].index(self.best_epoch)
                best_val_rmse = self.history['val_rmse'][best_idx]
                
                ax2.axvline(x=self.best_epoch, color='red', linestyle='--', 
                           label=f'Best Epoch ({self.best_epoch})', alpha=0.7)
                ax2.scatter([self.best_epoch], [best_val_rmse], 
                           color='red', s=150, zorder=5, marker='*')
                
                ax2.annotate(f'Best: {best_val_rmse:.4f}',
                           xy=(self.best_epoch, best_val_rmse),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('RMSE', fontsize=12)
            ax2.set_title('Validation RMSE', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        if self.history['val_rmse']:
            final_val_rmse = self.history['val_rmse'][-1]
            print("\n" + "="*60)
            print("Training Summary:")
            print("="*60)
            print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
            print(f"Final Val RMSE: {final_val_rmse:.4f}")
            print(f"Best Val RMSE: {self.best_val_rmse:.4f} (Epoch {self.best_epoch})")
            
            if final_val_rmse > self.best_val_rmse:
                diff = final_val_rmse - self.best_val_rmse
                pct = (diff / self.best_val_rmse) * 100
                print(f"Overfitting detected: +{diff:.4f} ({pct:.2f}%)")
            else:
                print("No overfitting detected")
            print("="*60)
