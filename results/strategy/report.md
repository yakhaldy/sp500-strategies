# SP500 ML Strategy — Report

## 1. Features used

Computed per ticker (`groupby('name')`) from the OHLCV data in `all_stocks_5yr.csv`, using the `ta` library:

- **RSI** (Relative Strength Index, 14-day window) — `rsi`
- **MACD** (12/26-day) — `macd`
- **Bollinger Bands** (20-day window) — `bb_high`, `bb_low`, and `bb_position` (the close price's relative position between the two bands)

Rows with NaN indicators (the first ~20 trading days of history per ticker, needed to warm up the rolling windows) are dropped. `APTV` is excluded: it only has 44 rows in the dataset (its history starts in Dec-2017), not enough for the 20-day rolling indicators.

### Target (no leakage)

On day D, the target is `sign(return(D+1, D+2))`, computed from `close` shifted by -1 and -2 within each ticker **before** the technical indicators are computed. Since RSI/MACD/Bollinger on row D only use information available up to close of day D, and the target on row D is intentionally the *future* return between D+1 and D+2, the feature/target alignment matches the schema required by the assignment with no forward-looking leakage.

## 2. Pipeline used

| Step | Choice |
|---|---|
| Imputer | `SimpleImputer(strategy="median")` — safety net; features have no NaN left after the feature-engineering drop, so this step is a no-op in practice |
| Scaler | `StandardScaler()` |
| Dimension reduction | **None.** Only 5 low-dimensional, already-interpretable handcrafted features. Adding PCA would replace `rsi`/`macd`/`bb_*` with opaque components, which conflicts with the required per-feature importance deliverable (`top_10_feature_importance.csv`) — so this step is intentionally omitted. |
| Model | `LGBMClassifier` |

The single canonical pipeline is defined once in `scripts/gridsearch.py::build_pipeline()` and reused identically by `model_selection.py` and `create_signal.py`, so the exact same pipeline is used for hyperparameter search, feature importance, and signal generation.

## 3. Cross-validation used

**Time Series Split (expanding window)** — `scripts/gridsearch.py::build_cv_splits()`.

- 10 folds, built on unique trading dates (never split mid-day across tickers).
- The first 2 years of train history (2013-03-18 → 2015-03-18) are fixed into every fold's training set; `sklearn.model_selection.TimeSeriesSplit` then expands the training window over the remaining train dates, fold by fold.
- Fold 1 (smallest) trains on **2013-03-18 → 2015-05-18 (~2.2 years, 264,545 rows)**, validates on 2015-05-19 → 2015-07-16.
- Fold 10 (largest) trains on 2013-03-18 → 2016-11-01 (446,980 rows), validates on 2016-11-02 → 2016-12-30 — strictly before the test set starts (2017-01-03), so the last training fold never touches the test period.

See `results/cross-validation/Time_series_split.png`.

## 4. Model selection

Grid search over `LGBMClassifier` hyperparameters (`n_estimators`, `max_depth`, `learning_rate`) on the 10 CV folds, picking the combination with the best **average validation AUC**:

- Best params: `n_estimators=100, max_depth=3, learning_rate=0.01`
- Average validation AUC (CV): **0.5175**
- Test AUC (2017+, never seen during CV): **0.4952**

See `results/cross-validation/ml_metrics_train.csv`, `metric_train.png`, `top_10_feature_importance.csv`, `results/selected-model/selected_model.txt`.

Predicting next-day stock direction from price-based technical indicators alone is close to a coin flip (AUC ≈ 0.50-0.52 across folds) — expected given the efficient-market literature and the small feature set. `bb_position` and `rsi` are consistently the two most important features across folds.

## 5. Machine learning signal

`scripts/create_signal.py` builds the signal with walk-forward validation, per the assignment's requirement that the pipeline is never trained once and applied everywhere:

- **Train period** (2015-05-19 → 2016-12-30): for each of the 10 folds, a fresh clone of the selected pipeline is fit on that fold's training rows only, and predicts `P(price up)` on that fold's validation rows. The 10 validation predictions are concatenated.
- **Test period** (2017-01-03 → 2018-02-05): the pipeline already fit once on the *entire* train set (`selected_model.pkl`) predicts on the test set — the natural continuation of the same expanding-window logic, using all information available before day D without ever seeing test data during training.

Result: `results/selected-model/ml_signal.csv`, a `(date, name)`-indexed series of `P(price increases between D+1 and D+2)`.

## 6. Strategy

**Binary long-only.** Each day D, go long $1 split evenly across every stock whose signal exceeds 0.5 that day (`weight = 1{signal>0.5} / n_active(D)`), so exactly $1 total is invested per day regardless of how many stocks are selected (0 invested on days with no signal above threshold).

Per the assignment's leakage warning, the position decided on day D (using information up to D) is multiplied by `return(D+1, D+2)` — the same value stored as `return_d1_d2` on row D — never by a return that overlaps with information used to build the signal.

### PnL

See `results/strategy/strategy.png` — cumulative PnL of the strategy vs. $1/day invested in the SP500 index, same axis, with a vertical line marking the train/test split (2017-01-01).

### Metrics (train vs. test)

| Set | Total PnL ($) | Max drawdown ($) | Annualized volatility | Sharpe ratio |
|---|---|---|---|---|
| Strategy — train | 0.113 | -0.171 | 0.156 | 0.44 |
| Strategy — test | 0.140 | -0.078 | 0.084 | 1.52 |
| SP500 — train | 0.068 | -0.141 | 0.147 | 0.28 |
| SP500 — test | 0.172 | -0.080 | 0.082 | 1.92 |

Full table: `results/strategy/results.csv`.

**Reading:** the strategy beats the SP500 on the in-sample train period ($0.113 vs $0.068), consistent with the (small but real) validation AUC edge picked up during CV. On the held-out test period it trails the index slightly ($0.140 vs $0.172) with a similar drawdown and volatility profile — consistent with the near-0.50 test AUC, i.e. the model's edge from technical indicators alone does not clearly generalize out-of-sample. This is a realistic and expected outcome for this type of simplified, weak-signal setup, not a training bug.
