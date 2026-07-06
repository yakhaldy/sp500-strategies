import joblib
import pandas as pd
from sklearn.base import clone

from features_engineering import load_data, features_engineering, prepare_final_dataset
from gridsearch import build_cv_splits, get_x_y


def build_ml_signal(folds, test, feature_cols, target_col, selected_pipeline):
    all_signals = []

    for fold_i, (train_fold, val_fold) in enumerate(folds):
        X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
        X_val, _ = get_x_y(val_fold, feature_cols, target_col)

        fold_pipeline = clone(selected_pipeline)
        fold_pipeline.fit(X_train, y_train)
        proba_val = fold_pipeline.predict_proba(X_val)[:, 1]

        signal_fold = pd.Series(proba_val, index=val_fold.index, name='ml_signal')
        all_signals.append(signal_fold)

        print(f"Fold {fold_i + 1}: signal built for {len(signal_fold)} rows")

    X_test, y_test = get_x_y(test, feature_cols, target_col)
    proba_test = selected_pipeline.predict_proba(X_test)[:, 1]
    signal_test = pd.Series(proba_test, index=test.index, name='ml_signal')
    all_signals.append(signal_test)
    print(f"Test period: signal built for {len(signal_test)} rows (pipeline fit on full train set)")

    ml_signal = pd.concat(all_signals).sort_index()
    return ml_signal


if __name__ == "__main__":
    stocks = load_data("data/all_stocks_5yr.csv")
    stocks = features_engineering(stocks)
    train, test, feature_cols, target_col = prepare_final_dataset(stocks)

    folds = build_cv_splits(train, n_splits=10)

    selected_pipeline = joblib.load("results/selected-model/selected_model.pkl")

    ml_signal = build_ml_signal(folds, test, feature_cols, target_col, selected_pipeline)

    print("\nShape signal:", ml_signal.shape)
    print(ml_signal.head())

    ml_signal.to_csv("results/selected-model/ml_signal.csv")
    print("Signal saved to results/selected-model/ml_signal.csv")
