from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_project_data, split_train_validation
from .metrics import evaluate_rmse
from .models.baselines import ItemAverageBaseline
from .predictions import save_predictions_to_csv


def describe_data(args: argparse.Namespace) -> None:
    train, test, movies = load_project_data(args.data_dir)
    print(f"Train ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")
    print(f"Movies: {len(movies):,}")
    print(f"Users: {train['user_id'].nunique():,}")
    print(f"Items in train: {train['item_id'].nunique():,}")


def baseline_predictions(args: argparse.Namespace) -> None:
    train, test, _ = load_project_data(args.data_dir)
    train_split, val_split = split_train_validation(train)
    model = ItemAverageBaseline().fit(train_split)
    print(f"Validation RMSE: {evaluate_rmse(model, val_split):.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions_to_csv(model, test, str(output), "ItemAverageBaseline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Movie Recommender utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser("describe-data", help="Print dataset sizes")
    describe.add_argument("--data-dir", default="data")
    describe.set_defaults(func=describe_data)

    baseline = sub.add_parser("baseline-predict", help="Fit item-average baseline and write test predictions")
    baseline.add_argument("--data-dir", default="data")
    baseline.add_argument("--output", default="outputs/preds/predictions_item_average.csv")
    baseline.set_defaults(func=baseline_predictions)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
