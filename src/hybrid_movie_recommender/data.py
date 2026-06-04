from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["item_id", "title", "genres", "description"]


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load a tab-separated ratings file with MovieLens-style columns."""
    return pd.read_csv(path, sep="\t", names=RATING_COLUMNS)


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load movie metadata used by the content-based recommender."""
    return pd.read_csv(path, sep="\t", names=MOVIE_COLUMNS)


def load_project_data(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train ratings, test ratings, and movie metadata from the project data directory."""
    data_dir = Path(data_dir)
    train = load_ratings(data_dir / "training.txt")
    test = load_ratings(data_dir / "test.txt")
    movies = load_movies(data_dir / "movies.txt")
    return train, test, movies


def split_train_validation(
    train_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 10,
    stratify_by_user: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the validation split used for hyperparameter sweeps."""
    stratify = train_data["user_id"] if stratify_by_user else None
    train_split, val_split = train_test_split(
        train_data, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True)
