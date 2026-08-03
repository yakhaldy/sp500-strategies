import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def build_cv_splits(train, n_splits=10, min_train_years=2):

    dates = train.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()

    start_date = unique_dates.min()
    min_train_end_date = start_date + pd.DateOffset(years=min_train_years)

    remaining_dates = unique_dates[unique_dates > min_train_end_date]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []

    print("\n==> Building CV splits...")

    for fold_i, (train_idx, val_idx) in enumerate(tscv.split(remaining_dates)):

        train_dates = remaining_dates[train_idx]
        val_dates = remaining_dates[val_idx]

        fixed_part = train[dates <= min_train_end_date]
        expanding_part = train[dates.isin(train_dates)]

        train_fold = pd.concat([fixed_part, expanding_part]).sort_index()
        val_fold = train[dates.isin(val_dates)]

        folds.append((train_fold, val_fold))

        print(
            f"Fold {fold_i + 1}: "
            f"Train {train_fold.index.get_level_values('date').min()} → "
            f"{train_fold.index.get_level_values('date').max()} "
            f"({len(train_fold)} rows), "
            f"Validation {val_fold.index.get_level_values('date').min()} → "
            f"{val_fold.index.get_level_values('date').max()} "
            f"({len(val_fold)} rows)"
        )

    return folds
    
    
def plot_cv_splits(
    folds,
    save_path="results/cross-validation/Time_series_split.png" ):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    for fold_i, (train_fold, val_fold) in enumerate(folds):

        train_dates = train_fold.index.get_level_values("date").unique()
        val_dates = val_fold.index.get_level_values("date").unique()

        ax.scatter(train_dates, [fold_i] * len(train_dates),
                   c='steelblue', s=40, label='Train' if fold_i == 0 else "")
        ax.scatter(val_dates, [fold_i] * len(val_dates),
                   c='orange', s=40, label='Validation' if fold_i == 0 else "")

    ax.set_title("Time Series Cross-Validation Splits")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fold")

    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([f"Fold {i+1}" for i in range(len(folds))])

    ax.legend()
    ax.invert_yaxis()

    fig.autofmt_xdate()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Image saved to: {save_path}")


def get_x_y(df, feature_cols, target_col):
    X = df[feature_cols]
    y = (df[target_col] == 1).astype(int)
    return X, y


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)),
    ])


param_grid = {
   'model__n_estimators': [100, 200],
   'model__max_depth': [3, 5],
   'model__learning_rate': [0.01, 0.1],
}
# param_grid = {
#     'model__n_estimators': [100, 200, 300],
#     'model__max_depth':    [3, 5, 7],
#     'model__learning_rate': [0.01, 0.05, 0.1],
#     'model__num_leaves':   [31, 63],       # LightGBM-specific
#     'model__min_child_samples': [20, 50],  # régularisation
# }



def evaluate_fold(pipeline, X, y):
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return {
        'auc': roc_auc_score(y, proba),
        'accuracy': accuracy_score(y, pred),
        'logloss': log_loss(y, proba),
    }


def grid_search_cv(folds, feature_cols, target_col, param_grid=param_grid):
    results = []
    best_score = -np.inf
    best_params = None
    best_fold_metrics = None

    for params in ParameterGrid(param_grid):
        fold_metrics = []
        for fold_i, (train_fold, val_fold) in enumerate(folds):
            X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
            X_val, y_val = get_x_y(val_fold, feature_cols, target_col)

            pipeline = build_pipeline()
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)

            train_metrics = evaluate_fold(pipeline, X_train, y_train)
            val_metrics = evaluate_fold(pipeline, X_val, y_val)

            fold_metrics.append({
                'fold': fold_i + 1,
                'train_auc': train_metrics['auc'], 'train_accuracy': train_metrics['accuracy'], 'train_logloss': train_metrics['logloss'],
                'val_auc': val_metrics['auc'], 'val_accuracy': val_metrics['accuracy'], 'val_logloss': val_metrics['logloss'],
            })

        avg_val_auc = sum(m['val_auc'] for m in fold_metrics) / len(fold_metrics)
        print(f"Params: {params}, Average Validation AUC: {avg_val_auc:.4f}")
        results.append({"params": params, "avg_val_auc": avg_val_auc, "fold_metrics": fold_metrics})

        if avg_val_auc > best_score:
            best_score = avg_val_auc
            best_params = params
            best_fold_metrics = fold_metrics

    return results, best_params, best_fold_metrics
