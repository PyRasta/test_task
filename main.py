import datetime
import os
import json
import multiprocessing
import threading

import websocket
import pandas as pd
import numpy as np

from binance import Client
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump, load

API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET')
CLIENT = Client(api_key=API_KEY, api_secret=API_SECRET)
CSV_DATASET = True

if os.path.exists('bitcoin_dependence_on_ether.joblib'):
    LEARNING = True
else:
    LEARNING = False

data_btc = []


def load_ethusdt_csv():
    dataframe_ethusdt = pd.read_csv('data/data/futures/um/daily/klines/ETHUSDT/1h/ETHUSDT20202023.csv')
    dataframe_ethusdt.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'x', 'c', 'a',
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


def get_klines_dataframe(start_klines_btcusdt, closes_ethusdt):
    if not LEARNING:
        if not CSV_DATASET:
            klines_start = [kline[:11] for kline in start_klines_btcusdt]
            klines = []
            for kline in klines_start:
                kline_temp = kline[1:6]
                klines.append(kline_temp)
            klines_float = []
            for kline_btcusdt, close_ethusdt in zip(klines, closes_ethusdt):
                kline_list = []
                for k in kline_btcusdt:
                    kline_list.append(float(k))
                kline_list.append(close_ethusdt)
                klines_float.append(kline_list)
            name_columns = [
                'open',
                'high',
                'low',
                'close',
                'volume',
                'close_ethusdt'
            ]
            klines_dataframe = pd.DataFrame(klines_float, columns=name_columns)
            return klines_dataframe
        else:
            dataframe_ethusdt = load_ethusdt_csv()
            dataframe_btcusdt = load_btcusdt_csv()
            closes_ethusdt = dataframe_ethusdt['close']
            dataframe_btcusdt['close_ethusdt'] = closes_ethusdt
            return dataframe_btcusdt

    else:
        name_columns = [
            'open',
            'high',
            'low',
            'close',
            'volume',
        ]
        klines = [start_klines_btcusdt]
        klines_dataframe = pd.DataFrame(klines, columns=name_columns)
        return klines_dataframe


def learn_model(dataframe):
    closes_ethusdt = dataframe['close_ethusdt']
    features = dataframe.drop('close_ethusdt', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, closes_ethusdt, test_size=0.2)
    y_test = np.nan_to_num(y_test)
    y_train = np.nan_to_num(y_train)
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    print('Train: ', regr.score(X_train, y_train))
    print('Test: ', regr.score(X_test, y_test))
    print('Intercept: ', regr.intercept_)
    dump(regr, 'bitcoin_dependence_on_ether.joblib')


def using_model(dataframe):
    model = load('bitcoin_dependence_on_ether.joblib')
    pred_price = model.predict(dataframe)
    return pred_price


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_message_btc(ws, message):
    global data_btc
    data = json.loads(message)
    data = data['k']
    data = [
        float(data['o']),
        float(data['h']),
        float(data['l']),
        float(data['c']),
        float(data['v']),
    ]
    data_btc = data
    names_thread = [i.name for i in threading.enumerate()]
    if not 'eth' in names_thread:
        watch_eth = WatchETH()
        ws_2 = websocket.WebSocketApp(
            f"wss://fstream.binance.com/ws/ethusdt_perpetual@continuousKline_1h",
            on_message=watch_eth.run,
            on_close=on_close
        )
        t_2 = threading.Thread(target=ws_2.run_forever, name='eth')
        t_2.start()


class WatchETH:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.start_price = None
        self.now_price = None

    def run(self, ws, message):
        data = json.loads(message)
        time_now = datetime.datetime.now()
        klines_dataframe = get_klines_dataframe(data_btc, None)
        pred_price = using_model(klines_dataframe)
        print(pred_price)
        res = abs(float(data['k']['c']) - pred_price)
        time_delta = time_now - self.start_time
        self.now_price = res
        if self.start_price is None or time_delta.seconds >= 60 * 60:
            self.start_price = res
        print(self.start_price, self.now_price)
        delta = abs(((self.start_price - self.now_price) / self.now_price * 100))
        if delta >= 1:
            print('ИЗМЕНЕНИЕ НА 1%')
        print(delta)


def run_tracking():
    ws_1 = websocket.WebSocketApp(
        f"wss://fstream.binance.com/ws/btcusdt_perpetual@continuousKline_1h",
        on_message=on_message_btc,
        on_close=on_close
    )
    t_1 = threading.Thread(target=ws_1.run_forever)
    t_1.start()


def main():
    if not LEARNING:
        klines_btcusdt = CLIENT.futures_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1500)
        klines_ethusdt = CLIENT.futures_klines(symbol="ETHUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=1500)
        closes_ethusdt = [float(kline[4]) for kline in klines_ethusdt]
        klines_dataframe = get_klines_dataframe(klines_btcusdt, closes_ethusdt)

        learn_model(klines_dataframe)
    run_tracking()


if __name__ == '__main__':
    main()
