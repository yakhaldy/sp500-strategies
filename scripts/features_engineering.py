import pandas as pd
import numpy as np
import ta 

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    return df

def data_info(data, file_path):
    print("=="*40 + "\n==> Data Info:" + file_path)
    print("Data Shape:", data.shape)
    print(data.head())
    print(data.dtypes)

    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")

    print("\nnumber of tickers:", len(data['name'].unique()))
    print("tickers:", data['name'].unique()[:10])

    print("\nfrom {} to {}".format(data['date'].min(), data['date'].max()))

    print("Missing values")
    print(data.isnull().sum())

    counts_per_ticker = data.groupby('name').size()
    print("\nmin days per ticker:", counts_per_ticker.min())
    print("max days per ticker:", counts_per_ticker.max())
    print("mean days per ticker:", counts_per_ticker.mean())

    print("\nnumber of tickers with less than 100 days:", counts_per_ticker[counts_per_ticker < 100])


def features_engineering(stocks):

    stocks['date'] = pd.to_datetime(stocks['date'], format="%Y-%m-%d")
    stocks = stocks.dropna()
    stocks = stocks[stocks['name'] != 'APTV']
    stocks = stocks.sort_values(['name', 'date']).reset_index(drop=True)

    close_d1 = stocks.groupby('name')['close'].shift(-1)
    close_d2 = stocks.groupby('name')['close'].shift(-2)

    stocks['return_d1_d2'] = (close_d2 - close_d1) / close_d1

    stocks['target'] = np.sign(stocks['return_d1_d2'])

    print("NAN values in target:", stocks['target'].isnull().sum())
    stocks = stocks.dropna(subset=['target'])
    stocks = stocks.reset_index(drop=True)

    # sample = stocks[stocks['name'] == 'AAL'].tail(5)
    # print(sample[['date', 'close', 'return_d1_d2', 'target']])

    #RSI
    stocks['rsi'] = (
    stocks.groupby('name')['close']
          .transform(lambda x: ta.momentum.rsi(x, window=14))
    )
    print("\nNAN values in rsi:", stocks['rsi'].isnull().sum())


    # MACD
    stocks['macd'] = (
    stocks.groupby('name')['close']
          .transform(lambda x: ta.trend.macd(x))
    )
    print("NAN values in macd:", stocks['macd'].isnull().sum())

    #Bollinger Bands
    stocks['bb_high'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.volatility.bollinger_hband(x, window=20))
    stocks['bb_low'] = stocks.groupby('name')['close'].transform(
        lambda x: ta.volatility.bollinger_lband(x, window=20))
    stocks['bb_position'] = (stocks['close'] - stocks['bb_low']) / (stocks['bb_high'] - stocks['bb_low'])

    print("\nbefore drop nan: Shape", stocks.shape)
    stocks = stocks.dropna(subset=['rsi', 'macd', 'bb_high', 'bb_low'])
    stocks = stocks.reset_index(drop=True)
    print("after drop nan: Shape", stocks.shape)

    return stocks




def prepare_final_dataset(stocks):
    stocks = stocks.set_index(['date', 'name']).sort_index()

    feature_cols = ['rsi', 'macd', 'bb_high', 'bb_low', 'bb_position']
    target_col = 'target'
    
    # Split
    dates = stocks.index.get_level_values('date')
    train = stocks[dates < '2017-01-01']
    test = stocks[dates >= '2017-01-01']
    
    return train, test, feature_cols, target_col







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

