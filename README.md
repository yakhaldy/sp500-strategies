# SP500 ML Strategy

Machine learning pipeline to generate a trading signal on S&P 500 constituents and backtest a strategy against the index.

## Approach

1. **Feature engineering** — Compute RSI, MACD, Stochastic, and Bollinger Bands per ticker from OHLCV data, with a 2-day forward return as target (no leakage: features on day D predict `return(D+1, D+2)`).
2. **Cross-validation** — Time Series Split with 10 expanding folds on the train set (pre-2017). Grid search over LightGBM hyperparameters.
3. **Signal generation** — Walk-forward: for each fold, the pipeline is trained on the fold's train set and predicts on its validation set. Predictions are concatenated to form the full ML signal.
4. **Strategy** — Binary long-only: go long $1 evenly distributed across stocks whose signal exceeds 0.5 each day.

## Project structure

```
sp500-strategies/
├── data/
│   ├── HistoricalData.csv       # SP500 index prices (benchmark)
│   └── all_stocks_5yr.csv       # SP500 constituents OHLCV (2013–2018)
├── scripts/
│   ├── features_engineering.py  # Feature computation + train/test split
│   ├── gridsearch.py            # CV splits + grid search utilities
│   ├── model_selection.py       # Model selection, metrics, plots
│   ├── create_signal.py         # Walk-forward ML signal generation
│   └── strategy.py              # Backtesting + PnL metrics
├── results/
│   ├── cross-validation/
│   │   ├── Time_series_split.png
│   │   ├── metric_train.png
│   │   ├── metric_train.csv
│   │   ├── ml_metrics_train.csv
│   │   └── top_10_feature_importance.csv
│   ├── selected-model/
│   │   ├── selected_model.pkl
│   │   ├── selected_model.txt
│   │   └── ml_signal.csv
│   └── strategy/
│       ├── strategy.png
│       ├── results.csv
│       └── report.md
└── requirements.txt
```

## How to run

All scripts must be run from the `scripts/` directory:

```bash
cd scripts

# 1. Model selection (grid search + CV metrics + plots) — ~10–20 min
python model_selection.py

# 2. Generate ML signal (walk-forward predictions)
python create_signal.py

# 3. Backtest the strategy
python strategy.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `ta`, `joblib`.

## Results summary

| Set | Total PnL ($) | Max drawdown | Sharpe |
|---|---|---|---|
| Strategy — train | 0.113 | -0.171 | 0.44 |
| Strategy — test  | 0.140 | -0.078 | 1.52 |
| SP500 — train    | 0.068 | -0.141 | 0.28 |
| SP500 — test     | 0.172 | -0.080 | 1.92 |

The strategy outperforms the SP500 in-sample (train AUC ~0.52 across folds). On the held-out test set (2017+) it trails slightly — consistent with near-random out-of-sample AUC (0.4952), which is expected for a weak-signal technical-indicator setup.
