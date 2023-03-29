import pandas as pd


def load_ethusdt_csv():
    dataframe_ethusdt = pd.read_csv('data/data/futures/um/daily/klines/ETHUSDT/1h/ETHUSDT20202023.csv')
    dataframe_ethusdt.columns = ['open_time', 'open', 'high', 'low', 'close_ethusdt', 'volume', 'close_time', 'x', 'c', 'a',
                                 'f', 'b']
    dataframe_ethusdt = dataframe_ethusdt.drop('open_time', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('close_time', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('x', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('c', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('a', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('f', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop('b', axis=1)
    dataframe_ethusdt = dataframe_ethusdt.drop_duplicates(keep=False)
    dataframe_ethusdt = dataframe_ethusdt.astype(float)
    return dataframe_ethusdt


def load_btcusdt_csv():
    dataframe = pd.read_csv('data/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT20202023.csv')
    dataframe.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'x', 'c', 'a',
                         'f', 'b']
    dataframe = dataframe.drop('open_time', axis=1)
    dataframe = dataframe.drop('close_time', axis=1)
    dataframe = dataframe.drop('x', axis=1)
    dataframe = dataframe.drop('c', axis=1)
    dataframe = dataframe.drop('a', axis=1)
    dataframe = dataframe.drop('f', axis=1)
    dataframe = dataframe.drop('b', axis=1)
    dataframe = dataframe.drop_duplicates(keep=False)
    dataframe = dataframe.astype(float)
    return dataframe


btc = load_btcusdt_csv()
eth = load_ethusdt_csv()
print(eth)
eth_close = eth['close_ethusdt']
btc['close_ethusdt'] = eth_close
print(btc)
