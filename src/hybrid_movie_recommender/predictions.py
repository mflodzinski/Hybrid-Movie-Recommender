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

from sklearn.metrics import mean_absolute_error, mean_squared_error

def save_predictions_to_csv(
    model,
    test_data: pd.DataFrame,
    output_filepath: str,
    model_name: str = "Model") -> pd.DataFrame:
    """
    Generate predictions for test data and save to CSV.
    """
    print(f"Generating predictions using {model_name}...")
    
    predictions_list = []
    
    for _, row in tqdm(test_data[['user_id', 'item_id', 'rating']].iterrows(), 
                       total=len(test_data), 
                       desc=f"{model_name} Predictions"):
        user_id = row['user_id']
        item_id = row['item_id']
        actual_rating = row['rating']
        
        predicted_rating = model.predict_rating(user_id, item_id)
        predicted_rating = float(np.clip(predicted_rating, 1.0, 5.0))
        
        predictions_list.append({
            'user_id': user_id,
            'item_id': item_id,
            'actual_rating': float(actual_rating),
            'predicted_rating': predicted_rating
        })
    
    predictions_df = pd.DataFrame(predictions_list)
    
    rmse = np.sqrt(mean_squared_error(predictions_df['actual_rating'], 
                                       predictions_df['predicted_rating']))
    mae = mean_absolute_error(predictions_df['actual_rating'], 
                               predictions_df['predicted_rating'])
    
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_filepath, index=False)
    
    print(f"\nPredictions saved to: {output_filepath}")
    print(f"Results for {model_name}:")
    print(f"   - Total predictions: {len(predictions_df)}")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")

    return predictions_df
