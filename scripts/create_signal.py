import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from features_engineering import load_data, features_engineering, prepare_final_dataset
from gridsearch import build_cv_splits


def get_x_y(df, feature_cols, target_col):
    X = df[feature_cols]
    y = (df[target_col] == 1).astype(int)
    return X, y


def build_ml_signal(folds, feature_cols, target_col, best_params):
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
    ])
    pipeline.set_params(**best_params)
    
    all_signals = []
    
    for fold_i, (train_fold, val_fold) in enumerate(folds):
        X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
        X_val, y_val = get_x_y(val_fold, feature_cols, target_col)
        
        pipeline.fit(X_train, y_train)
        proba_val = pipeline.predict_proba(X_val)[:, 1]
        
        signal_fold = pd.Series(proba_val, index=val_fold.index, name='ml_signal')
        all_signals.append(signal_fold)
        
        print(f"Fold {fold_i+1}: signal built for {len(signal_fold)} rows")
    
    ml_signal = pd.concat(all_signals).sort_index()
    return ml_signal


if __name__ == "__main__":
    stocks = load_data("data/all_stocks_5yr.csv")
    stocks = features_engineering(stocks)
    train, test, feature_cols, target_col = prepare_final_dataset(stocks)
    
    folds = build_cv_splits(train, n_splits=10)
    
    best_params = {
        'model__learning_rate': 0.01,
        'model__max_depth': 3,
        'model__n_estimators': 100
    }
    
    ml_signal = build_ml_signal(folds, feature_cols, target_col, best_params)
    
    print("\nShape signal:", ml_signal.shape)
    print(ml_signal.head())
    
    ml_signal.to_csv("results/selected-model/ml_signal.csv")
    print("✅ Signal saved to results/selected-model/ml_signal.csv")