import itertools
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss



from features_engineering import load_data, data_info, features_engineering, prepare_final_dataset
from gridsearch import build_cv_splits, plot_cv_splits, grid_search_cv, get_x_y




pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier(random_state=42, verbose=-1,  n_jobs=-1))
])








def evaluate_model(train, test, feature_cols, target_col, best_params):
    pipeline.set_params(**best_params)

    x_train, y_train = get_x_y(train, feature_cols, target_col) 
    x_test, y_test = get_x_y(test, feature_cols, target_col)

    pipeline.fit(x_train, y_train)                    
    proba_test = pipeline.predict_proba(x_test)[:, 1] 
    pred_test = pipeline.predict(x_test)

    auc = roc_auc_score(y_test, proba_test)
    acc = accuracy_score(y_test, pred_test)
    loss = log_loss(y_test, proba_test)

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test LogLoss: {loss:.4f}")

    joblib.dump(pipeline, "results/selected-model/selected_model.pkl")

    return {'auc': auc, 'accuracy': acc, 'logloss': loss}






def compute_feature_importance_per_fold(folds, feature_cols, target_col, best_params):
    pipeline.set_params(**best_params)
    all_importance = []
    
    for fold_i, (train_fold, val_fold) in enumerate(folds):
        X_train, y_train = get_x_y(train_fold, feature_cols, target_col)
        pipeline.fit(X_train, y_train)
        
        fold_importance = get_feature_importance(pipeline, feature_cols, fold_i + 1)
        all_importance.append(fold_importance)
    
    result = pd.concat(all_importance, ignore_index=True)
    result.to_csv("results/cross-validation/top_10_feature_importance.csv", index=False)
    print("Feature importance saved")
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

import pandas as pd

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

def get_feature_importance(pipeline, feature_cols, fold_num):
    model = pipeline.named_steps['model']
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        'fold': fold_num,
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return df.head(10)

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

    results, best_params, best_fold_metrics = grid_search_cv(pipeline, folds, feature_cols, target_col)
    evaluate_model(train, test, feature_cols, target_col, best_params)
    best_result = max(results, key=lambda r: r['avg_val_auc'])
    save_model_info(pipeline, best_params, best_result['avg_val_auc'])
    save_ml_metrics(best_fold_metrics)
    compute_feature_importance_per_fold(folds, feature_cols, target_col, best_params)