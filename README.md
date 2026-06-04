# Hybrid Movie Recommender

A recommender-systems project for movie rating prediction and top-k ranking. The original notebook implementation has been refactored into reusable Python modules while keeping the experiment notebook as a readable report.

## Project Structure

```text
data/                         # Small input datasets tracked in Git
notebooks/                    # Narrative experiment notebook
src/hybrid_movie_recommender/  # Importable recommender package
outputs/                      # Ignored generated predictions, plots, and saved models
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a lightweight CLI command from the project root:

```bash
PYTHONPATH=src python -m hybrid_movie_recommender.cli describe-data
PYTHONPATH=src python -m hybrid_movie_recommender.cli baseline-predict
```

## Implemented Models

- Content-based collaborative filtering using TF-IDF movie text features, optional truncated SVD, normalized item embeddings, and user profiles built from rated movies.
- User-based and item-based k-nearest-neighbor collaborative filtering with similarity matrices and top-k recommendation methods.
- Matrix factorization trained with PyTorch SGD, user/item embeddings, optional bias terms, L2 regularization, and validation tracking.
- Bayesian Personalized Ranking trained with sampled user-positive-negative triples for ranking-oriented embeddings.
- Rating hybrid model that learns Ridge-regression weights over component model predictions.
- Ranking hybrid model that optimizes per-model score weights for NDCG@10 using Dirichlet search and coordinate refinement.
- Baselines for item-average rating prediction, mean hybrid rating prediction, random ranking, and popularity ranking.

## Results

Rating prediction:

| Model | RMSE | MAE |
| :-- | --: | --: |
| Content-Based | 1.1762 | 0.9111 |
| Bayesian Personalized Ranking (BPR) | 1.1554 | 0.9063 |
| Item Average Baseline | 1.0331 | 0.8277 |
| Item-Based CF | 1.0064 | 0.7863 |
| User-Based CF | 0.9675 | 0.7564 |
| Matrix Factorization (MF) | 0.9403 | 0.7427 |
| Mean Hybrid Baseline | 0.9399 | 0.7393 |
| **Hybrid Model (Rating)** | **0.9250** | **0.7285** |

Ranking at k=10:

| Model | Precision@10 | Recall@10 | NDCG@10 |
| :-- | --: | --: | --: |
| Random Recommender | 0.0161 | 0.0067 | 0.0114 |
| Content-Based | 0.1046 | 0.0386 | 0.0955 |
| User-Based CF | 0.0107 | 0.0023 | 0.1041 |
| Matrix Factorization (MF) | 0.0791 | 0.0225 | 0.1443 |
| Item-Based CF | 0.0266 | 0.0030 | 0.1627 |
| Bayesian Personalized Ranking (BPR) | 0.2137 | 0.0930 | 0.1778 |
| Popularity Recommender | 0.2231 | 0.1048 | 0.1842 |
| **Hybrid Model (Ranking)** | **0.3194** | **0.1224** | **0.3100** |

## Notes