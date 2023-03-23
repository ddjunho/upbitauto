import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import schedule
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from upbit_keys import access, secret

buy_unit = 0.1   # 분할 매수 금액 단위 설정
k = 0
COIN = "KRW-BTC" #코인명
day_s = 0  #15*96은 1일

def vola_break(ticker):
    # 변동성 돌파 전략
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    vola_break_price = (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return vola_break_price
vola_break_price = vola_break(COIN)
    
def get_target_price(ticker): #매수최저가예측
    # 데이터 불러오기
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=192)
    # 입력 데이터 전처리
    X = df[['open', 'high', 'low', 'close', 'volume']].values  # 입력 데이터는 open, high, low, close, volume 5가지 종류
    X_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)
    # 출력 데이터 전처리
    y = df['low'].values  # 출력 데이터는 high 가격
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))
    # 학습 데이터 생성
    X_train = []
    y_train = []
    for i in range(192, len(X)):
        X_train.append(X[i - 192:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(192, 5)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse')
    # 학습
    model.fit(X_train, y_train, epochs=100, verbose=0)
    # 새로운 데이터에 대한 예측
    last_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-192:].values  # 가장 최근 192개 데이터
    last_data = X_scaler.transform(last_data.reshape((1, -1, 5)))  # 입력 데이터 전처리
    predicted_price = model.predict(last_data)  # 예측 결과
    predicted_price = y_scaler.inverse_transform(predicted_price)
    return predicted_price + vola_break_price
    
def get_balance(ticker):
    # 잔고 조회
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    # 현재가 조회
    try:
        return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]
    except:
        return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["bid_price"]
    
def predict_sell_price(ticker):
    # 데이터 불러오기
    df = pyupbit.get_ohlcv(ticker, interval="minute15", count=192)
    # 입력 데이터 전처리
    X = df[['open', 'high', 'low', 'close', 'volume']].values  # 입력 데이터는 open, high, low, close, volume 5가지 종류
    X_scaler = MinMaxScaler()
    X = X_scaler.fit_transform(X)
    # 출력 데이터 전처리
    y = df['high'].values  # 출력 데이터는 high 가격
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))
    # 학습 데이터 생성
    X_train = []
    y_train = []
    for i in range(192, len(X)):
        X_train.append(X[i - 192:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(192, 5)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse')
    # 학습
    model.fit(X_train, y_train, epochs=100, verbose=0)
    # 새로운 데이터에 대한 예측
    last_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-192:].values  # 가장 최근 192개 데이터
    last_data = X_scaler.transform(last_data.reshape((1, -1, 5)))  # 입력 데이터 전처리
    predicted_price = model.predict(last_data)  # 예측 결과
    predicted_price = y_scaler.inverse_transform(predicted_price)
    return predicted_price - vola_break_price
    
# 로그인
upbit = pyupbit.Upbit(access, secret)

# 자동매매 시작 함수
krw = get_balance("KRW")
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
def run_auto_trade():
    while True:
        try:
            now = datetime.datetime.now()
            target_price = get_target_price(COIN)
            predicted_sell_price = predict_sell_price(COIN)
            current_price = get_current_price(COIN)
            if now.hour == 9 and now.minute == 0:
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
                
            if target_price > current_price and target_price < predicted_sell_price:
                if get_balance("KRW") < krw * buy_unit:
                    buy_amount = krw * 0.9995
                upbit.buy_market_order(COIN, buy_amount)
            else:
                if current_price >= predicted_sell_price:
                    btc = get_balance("BTC")
                    if btc > 0.00008:
                        sell_amount = btc * 1
                        upbit.sell_market_order(COIN, sell_amount)
        except Exception as e:
            print(e)
            time.sleep(1)
# 스케줄러 설정
schedule.every(1).seconds.do(run_auto_trade)
print("autotrade start")

# 스케줄러 실행
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
