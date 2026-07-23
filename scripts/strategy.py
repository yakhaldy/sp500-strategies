import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features_engineering import load_data, features_engineering

TEST_START = '2017-01-01'


def load_ml_signal(path='results/selected-model/ml_signal.csv'):
    signal = pd.read_csv(path, parse_dates=['date']).set_index(['date', 'name'])['ml_signal']
    return signal.sort_index()


def load_returns():
    stocks = load_data('data/all_stocks_5yr.csv')
    stocks = features_engineering(stocks)
    stocks = stocks.set_index(['date', 'name']).sort_index()
    return stocks['return_d1_d2']


def build_binary_long_only_strategy(ml_signal, threshold=0.5):
    signal_binary = (ml_signal > threshold).astype(float)
    n_active = signal_binary.groupby(level='date').transform('sum')
    weights = signal_binary.where(n_active == 0, signal_binary / n_active)
    return weights


def compute_daily_pnl(weights, returns):
    df = pd.concat([weights.rename('weight'), returns.rename('return')], axis=1).dropna()
    df['pnl'] = df['weight'] * df['return']
    return df.groupby(level='date')['pnl'].sum().sort_index()


def load_benchmark_daily_pnl(dates, path='data/HistoricalData.csv'):
    sp500 = pd.read_csv(path)
    sp500.columns = sp500.columns.str.strip().str.lower()
    sp500['date'] = pd.to_datetime(sp500['date'], format='%m/%d/%y')
    sp500 = sp500.sort_values('date').set_index('date')
    daily_return = sp500['close'].pct_change()
    return daily_return.reindex(dates).fillna(0.0)


def max_drawdown(cum_pnl):
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    return drawdown.min()


def compute_metrics(daily_pnl, label):
    cum_pnl = daily_pnl.cumsum()
    total_pnl = cum_pnl.iloc[-1] if len(cum_pnl) else 0.0
    volatility = daily_pnl.std() * np.sqrt(252)
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else np.nan
    return {
        'set': label,
        'n_days': len(daily_pnl),
        'total_pnl': total_pnl,
        'max_drawdown': max_drawdown(cum_pnl),
        'volatility_annualized': volatility,
        'sharpe_ratio': sharpe,
    }


def plot_strategy(strategy_cum, benchmark_cum, test_start, save_path='results/strategy/strategy.png'):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(strategy_cum.index, strategy_cum.values, label='ML strategy PnL', color='steelblue')
    ax.plot(benchmark_cum.index, benchmark_cum.values, label='SP500 PnL ($1 invested/day)', color='darkorange')
    ax.axvline(pd.Timestamp(test_start), color='grey', linestyle='--', label='Train / Test split')

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative PnL ($)')
    ax.set_title('ML Strategy vs SP500 - Cumulative PnL')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    ml_signal = load_ml_signal()
    returns = load_returns()

    weights = build_binary_long_only_strategy(ml_signal)
    daily_pnl = compute_daily_pnl(weights, returns)
    benchmark_pnl = load_benchmark_daily_pnl(daily_pnl.index)

    strategy_cum = daily_pnl.cumsum()
    benchmark_cum = benchmark_pnl.cumsum()

    plot_strategy(strategy_cum, benchmark_cum, TEST_START)

    train_mask = daily_pnl.index < TEST_START
    test_mask = ~train_mask

    results = [
        compute_metrics(daily_pnl[train_mask], 'strategy_train'),
        compute_metrics(daily_pnl[test_mask], 'strategy_test'),
        compute_metrics(benchmark_pnl[train_mask], 'sp500_train'),
        compute_metrics(benchmark_pnl[test_mask], 'sp500_test'),
    ]
    results_df = pd.DataFrame(results).set_index('set')
    results_df.to_csv('results/strategy/results.csv')

    print("\n" + results_df.to_string())
    print("\nResults saved to results/strategy/results.csv")
