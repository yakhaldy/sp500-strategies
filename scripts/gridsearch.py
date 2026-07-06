from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


def time_series_split(train, n_splits=10):
    unique_dates = train.index.get_level_values('date').unique()
    start_date = unique_dates.min()
    max_date = unique_dates.max()
    min_train_years = 2
    min_train_end_date = start_date + pd.DateOffset(years=min_train_years)
    remaining_dates = unique_dates[unique_dates > min_train_end_date]
    time_series_split = TimeSeriesSplit(n_splits=n_splits)

    folds = []
    print("\n==> Building CV splits...")
    for fold_i, (train_idx, val_idx) in enumerate(time_series_split.split(remaining_dates)):
        train_dates = remaining_dates[train_idx]
        val_dates = remaining_dates[val_idx]

        fixed_part = train[train.index.get_level_values('date') < min_train_end_date]
        expanding_part = train[train.index.get_level_values('date').isin(train_dates)]

        train_fold = pd.concat([fixed_part, expanding_part]).sort_index()
        val_fold = train[train.index.get_level_values('date').isin(val_dates)]

        folds.append((train_fold, val_fold))
        print("Fold {}: Train from {} to {}, Val from {} to {}".format(
            fold_i + 1,
            train_fold.index.get_level_values('date').min(),
            train_fold.index.get_level_values('date').max(),
            val_fold.index.get_level_values('date').min(),
            val_fold.index.get_level_values('date').max()
        ))

    return folds


param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1]
}


def plot_cv_splits(folds, save_path='results/cross-validation/Time_series_split.png'):
    fig, ax = plt.subplots(figsize=(14, 6))

    for fold_i, (train_fold, val_fold) in enumerate(folds):
        train_dates = train_fold.index.get_level_values('date').unique()
        val_dates = val_fold.index.get_level_values('date').unique()

        ax.scatter(train_dates, [fold_i] * len(train_dates), 
                       c='steelblue', s=40, label='Train' if fold_i == 0 else "")
        ax.scatter(val_dates, [fold_i] * len(val_dates), 
                       c='orange', s=40, label='Validation' if fold_i == 0 else "")

        ax.set_yticks(range(len(folds)))
        ax.set_yticklabels([f'Fold {i+1}' for i in range(len(folds))])
        ax.set_xlabel('Date')
        ax.set_ylabel('Fold')
        ax.set_title('Time Series Cross-Validation Splits')
        ax.legend(loc='upper left')
        ax.invert_yaxis()  

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Image saved to: {save_path}")





def get_x_y(df, feature_cols, target_col):
    X = df[feature_cols]
    y = (df[target_col] == 1).astype(int)
    return X, y


##############################################
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
RANDOM_STATE = 42
##############################################
def make_pipeline(model_name="rf"):
    if model_name == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)
    else:
        clf = LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_STATE)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ], memory=None)
################################################

def evaluate_fold(pipeline, X, y):
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return {
        'auc': roc_auc_score(y, proba),
        'accuracy': accuracy_score(y, pred),
        'logloss': log_loss(y, proba)
    }

def grid_search_cv(folds, feature_cols, target_col):
    model_name = ["rf", "gb", "lr"]
    best_auc = -np.inf
    beast_model_name = None
    best_params = None
    best_fold_metrics = None
    results = []
    for mname in model_name:
        fold_metrics = []
        for fold_i, (train_fold, val_fold) in enumerate(folds):
                X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
                X_val, y_val = get_x_y(val_fold, feature_cols, target_col)
                pipe = make_pipeline(mname)
                pipe.fit(X_train, y_train)
                train_metrics = evaluate_fold(pipe, X_train, y_train)
                val_metrics = evaluate_fold(pipe, X_val, y_val)
                fold_metrics.append({
                    'fold': fold_i + 1,
                    'train_auc': train_metrics['auc'], 'train_accuracy': train_metrics['accuracy'], 'train_logloss': train_metrics['logloss'],
                    'val_auc': val_metrics['auc'], 'val_accuracy': val_metrics['accuracy'], 'val_logloss': val_metrics['logloss'],
                })
                print(f"Model: {mname}, Fold {fold_i + 1}: Train AUC: {train_metrics['auc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        avg_val_auc = sum(m['val_auc'] for m in fold_metrics) / len(fold_metrics)
        print(f"Model: {mname}, Average Validation AUC: {avg_val_auc:.4f}")
        if avg_val_auc > best_auc:
            best_auc = avg_val_auc
            best_model_name = mname
            best_params = pipe.get_params()
            best_fold_metrics = fold_metrics
        results.append({"params": best_params, "avg_val_auc": avg_val_auc, "fold_metrics": fold_metrics})
        return results, best_params, best_fold_metrics


    # results = []
    # best_score = float("-inf")
    # best_params = None
    # best_fold_metrics = None

    # for params in ParameterGrid(param_grid):
    #     pipeline.set_params(**params)
    #     fold_metrics = []

    #     for fold_i, (train_fold, val_fold) in enumerate(folds):
    #         X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
    #         X_val, y_val = get_x_y(val_fold, feature_cols, target_col)

    #         pipeline.fit(X_train, y_train)

    #         train_metrics = evaluate_fold(pipeline, X_train, y_train)
    #         val_metrics = evaluate_fold(pipeline, X_val, y_val)

    #         fold_metrics.append({
    #             'fold': fold_i + 1,
    #             'train_auc': train_metrics['auc'], 'train_accuracy': train_metrics['accuracy'], 'train_logloss': train_metrics['logloss'],
    #             'val_auc': val_metrics['auc'], 'val_accuracy': val_metrics['accuracy'], 'val_logloss': val_metrics['logloss'],
    #         })

    #     avg_val_auc = sum(m['val_auc'] for m in fold_metrics) / len(fold_metrics)
    #     results.append({"params": params, "avg_val_auc": avg_val_auc, "fold_metrics": fold_metrics})

    #     if avg_val_auc > best_score:
    #         best_score = avg_val_auc
    #         best_params = params
    #         best_fold_metrics = fold_metrics

    # return results, best_params, best_fold_metrics



if __name__ == "__main__":
    grid_search_cv(make_pipeline("rf"), [], [], [])

