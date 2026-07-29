import pandas as pd
import numpy as np
import ta 

import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def data_info(data, file_path):
    print("=" * 60)
    print("Data Info:", file_path)
    print("=" * 60)

    print("\nShape:")
    print(data.shape)

    print("\nHead:")
    print(data.head())

    print("\nData Types:")
    print(data.dtypes)

    if "date" in data.columns:
        data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")

        print("\nDate range:")
        print(data["date"].min(), "to", data["date"].max())

    if "name" in data.columns:
        print("\nNumber of tickers:")
        print(data["name"].nunique())

        print("\nTickers:")
        print(data["name"].unique()[:10])

        counts = data.groupby("name").size()

        print("\nDays per ticker:")
        print("Min:", counts.min())
        print("Max:", counts.max())
        print("Mean:", counts.mean())

    print("\nMissing values:")
    print(data.isnull().sum())


def features_engineering(stocks):

    stocks['date'] = pd.to_datetime(stocks['date'])

    stocks = stocks.dropna(
        subset=['date','name','close']
    )

    stocks = stocks[
        stocks['name'] != 'APTV'
    ]

    stocks = stocks.sort_values(
        ['name','date']
    ).reset_index(drop=True)


    close_d1 = stocks.groupby('name')['close'].shift(-1)
    close_d2 = stocks.groupby('name')['close'].shift(-2)

    stocks['return_d1_d2'] = (
        close_d2 - close_d1
    ) / close_d1

    stocks['target'] = np.sign(
        stocks['return_d1_d2']
    )

    stocks = stocks.dropna(
        subset=['target']
    )


    # RSI
    stocks['rsi'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.momentum.rsi(x, window=14)
    )


    # MACD
    stocks['macd'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.trend.macd(x)
    )


    # Bollinger
    stocks['bb_high'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.volatility.bollinger_hband(x, window=20)
    )

    stocks['bb_low'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.volatility.bollinger_lband(x, window=20)
    )


    stocks['bb_position'] = (
        (stocks['close'] - stocks['bb_low']) /
        (stocks['bb_high'] - stocks['bb_low'])
    )


    stocks = stocks.replace(
        [np.inf, -np.inf],
        np.nan
    )


    stocks = stocks.dropna().reset_index(drop=True)


    return stocks


def prepare_final_dataset(stocks):

    stocks = stocks.set_index(["date", "name"]).sort_index()

    feature_cols = [
        "rsi",
        "macd",
        "bb_high",
        "bb_low",
        "bb_position",
    ]

    target_col = "target"

    split_date = "2017-01-01"

    dates = stocks.index.get_level_values("date")

    train = stocks[dates < split_date]
    test = stocks[dates >= split_date]

    return train, test, feature_cols, target_col
