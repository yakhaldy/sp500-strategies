import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone

from features_engineering import load_data, data_info, features_engineering, prepare_final_dataset
from gridsearch import build_cv_splits, plot_cv_splits, build_pipeline, grid_search_cv, get_x_y, evaluate_fold


def evaluate_model(pipeline, train, test, feature_cols, target_col):
    x_train, y_train = get_x_y(train, feature_cols, target_col)
    x_test, y_test = get_x_y(test, feature_cols, target_col)

    pipeline.fit(x_train, y_train)
    test_metrics = evaluate_fold(pipeline, x_test, y_test)

    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test LogLoss: {test_metrics['logloss']:.4f}")

    joblib.dump(pipeline, "results/selected-model/selected_model.pkl")
    print("Selected model saved to results/selected-model/selected_model.pkl")

    return pipeline, test_metrics


def get_feature_importance(pipeline, feature_cols, fold_num):
    model = pipeline.named_steps['model']
    importances = model.feature_importances_

    df = pd.DataFrame({
        'fold': fold_num,
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return df.head(10)


def compute_feature_importance_per_fold(folds, feature_cols, target_col, best_params):
    all_importance = []

    for fold_i, (train_fold, val_fold) in enumerate(folds):
        X_train, y_train = get_x_y(train_fold, feature_cols, target_col)

        pipeline = clone(build_pipeline())
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)

        fold_importance = get_feature_importance(pipeline, feature_cols, fold_i + 1)
        all_importance.append(fold_importance)

    result = pd.concat(all_importance, ignore_index=True)
    result.to_csv("results/cross-validation/top_10_feature_importance.csv", index=False)
    print("Feature importance saved to results/cross-validation/top_10_feature_importance.csv")
    return result


def save_model_info(pipeline, best_params, avg_val_auc, save_path="results/selected-model/selected_model.txt"):
    with open(save_path, 'w') as f:
        f.write("Selected Model: LGBMClassifier\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Hyperparameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")

        f.write(f"\nAverage Validation AUC (CV): {avg_val_auc:.4f}\n")

        f.write("\nFull Pipeline Steps:\n")
        for step_name, step_obj in pipeline.steps:
            f.write(f"  - {step_name}: {step_obj}\n")

    print(f"Model info saved to {save_path}")


def save_ml_metrics(best_fold_metrics, save_path="results/cross-validation/ml_metrics_train.csv"):
    rows = []
    for fm in best_fold_metrics:
        rows.append({'fold': fm['fold'], 'set': 'train', 'auc': fm['train_auc'],
                     'accuracy': fm['train_accuracy'], 'logloss': fm['train_logloss']})
        rows.append({'fold': fm['fold'], 'set': 'validation', 'auc': fm['val_auc'],
                     'accuracy': fm['val_accuracy'], 'logloss': fm['val_logloss']})

    df = pd.DataFrame(rows).set_index(['fold', 'set'])
    df.to_csv(save_path)
    print(f"Metrics saved to {save_path}")
    return df


def plot_metric_train(ml_metrics_df, metric='auc', save_path='results/cross-validation/metric_train.png'):
    
    pivot = ml_metrics_df.reset_index().pivot(index='fold', columns='set', values=metric)

    folds = pivot.index.tolist()
    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, pivot['train'], width, label='Train set', color='lightgrey', edgecolor='black')
    ax.bar(x + width / 2, pivot['validation'], width, label='Validation set', color='olivedrab', edgecolor='black')

    ax.set_xlabel('Fold')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'AUC on train and validation set\non all folds of the train set')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Metric plot saved to {save_path}")


if __name__ == "__main__":
    stocks = load_data("data/all_stocks_5yr.csv")
    data_info(stocks, "data/all_stocks_5yr.csv")
    stocks = features_engineering(stocks)

    train, test, feature_cols, target_col = prepare_final_dataset(stocks)
    print("\ntrain from {} to {}".format(train.index.get_level_values('date').min(), train.index.get_level_values('date').max()))
    print("test from {} to {}".format(test.index.get_level_values('date').min(), test.index.get_level_values('date').max()))
    print("\nTrain shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nFeatures:", feature_cols)

    folds = build_cv_splits(train, n_splits=10)
    plot_cv_splits(folds, save_path='results/cross-validation/Time_series_split.png')

    results, best_params, best_fold_metrics = grid_search_cv(folds, feature_cols, target_col)
    best_result = max(results, key=lambda r: r['avg_val_auc'])
    print(f"\nBest params: {best_params} (avg val AUC: {best_result['avg_val_auc']:.4f})")

    selected_pipeline = build_pipeline()
    selected_pipeline.set_params(**best_params)
    selected_pipeline, test_metrics = evaluate_model(selected_pipeline, train, test, feature_cols, target_col)

    save_model_info(selected_pipeline, best_params, best_result['avg_val_auc'])
    ml_metrics_df = save_ml_metrics(best_fold_metrics)
    plot_metric_train(ml_metrics_df, metric='auc')
    compute_feature_importance_per_fold(folds, feature_cols, target_col, best_params)
