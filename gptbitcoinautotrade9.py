import time
import json
import pyupbit
import pandas as pd
import numpy as np
import schedule
import tensorflow as tf
import requests.exceptions
import simplejson.errors
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from upbit_keys import access, secret
tf.config.run_functions_eagerly(True)
buy_unit = 0.1   # 분할 매수 금액 단위 설정
k = 0.1
COIN = "KRW-BTC" #코인명
day_s = 0  #15*96은 1일

def vola_break(ticker):
    # 변동성 돌파 전략
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    vola_break_price = (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return vola_break_price
vola_break_price = vola_break(COIN)

def get_balance(ticker):
    # 원화 잔고 조회
    try:
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == ticker:
                if b['balance'] is not None:
                    return float(b['balance'])
                else:
                    return 0
        # 해당 티커의 잔고가 없을 경우 0을 반환
        return 0
    except (requests.exceptions.RequestException, simplejson.errors.JSONDecodeError) as e:
        print(f"에러 발생: {e}")
    return 0
def get_current_price(ticker):
    # 현재가 조회
    try:
        return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]
    except:
        return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["bid_price"]
    
def predict_target_price(target_type):
    with open(f"{target_type}.json") as f:
        input_data = json.load(f)

    ticker = input_data['arguments']['ticker']
    target_type = input_data['arguments']['target_type']

    # 데이터 불러오기
    df1 = pyupbit.get_ohlcv(ticker, interval="day", count=183)
    df2 = pyupbit.get_ohlcv(ticker, interval="day", count=183, to=df1.index[0])
    df = pd.concat([df2, df1])
    # 입력 데이터 전처리
    X = df[['open', 'high', 'low', 'close', 'volume']].values
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    # 출력 데이터 전처리
    y = df[target_type].values
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))
    # 학습 데이터 생성
    X_train = []
    y_train = []
    for i in range(365, len(X)):
        X_train.append(X[i - 365:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(365, 5)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    # 학습
    model.fit(X_train, y_train, epochs=100, verbose=1)
    # 새로운 데이터에 대한 예측
    last_data = df[['open', 'high', 'low', 'close', 'volume']].iloc[-365:].values
    last_data_mean = last_data.mean(axis=0)
    last_data_std = last_data.std(axis=0)
    last_data = (last_data - last_data_mean) / last_data_std
    # 예측할 데이터의 shape를 (1, 365, 5)로 변경
    last_data = np.expand_dims(last_data, axis=0)
    predicted_price = model.predict(last_data)
    predicted_price = y_scaler.inverse_transform(predicted_price)
    predicted_price = ['{:.5f}'.format(p) for p in predicted_price.flatten()]
    predicted_price = [[float(p)] for p in predicted_price]
    if target_type == 'low':
        return predicted_price + vola_break_price
    elif target_type == 'high':
        return predicted_price - vola_break_price
    else:
        raise ValueError('Invalid target type. Choose "low" or "high".')
# 로그인
upbit = pyupbit.Upbit(access, secret)
krw = get_balance("KRW")
target_price = predict_target_price("low")
predicted_sell_price = predict_target_price("high")
current_price = get_current_price(COIN)
btc = get_balance(COIN)
# 자동매매 시작 함수
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
def run_auto_trade():
    global target_price 
    global predicted_sell_price
    global current_price
    global buy_amount
    global krw
    global btc
    while True:
        try:
            now = datetime.now()
            current_price = get_current_price(COIN)
            if now.hour == 9 and now.minute == 0 :
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
                target_price = predict_target_price(COIN, 'low')
                predicted_sell_price = predict_target_price(COIN, 'high')
            if krw is not None and target_price - vola_break_price <= current_price and target_price >= current_price and target_price < predicted_sell_price and target_price > 100:
                if get_balance("KRW") < krw * buy_unit:
                    buy_amount = krw * 0.9995
                upbit.buy_market_order(COIN, buy_amount)
                print(now, "매수")
            else:
                if btc != 0 and btc is not None and current_price >= predicted_sell_price:
                    btc = get_balance(COIN)
                    sell_amount = btc
                    upbit.sell_market_order(COIN, sell_amount)
                    print(now, "매도")
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(1)
# 스케줄러 설정
schedule.every(1).seconds.do(run_auto_trade)
print("매수가 조회 :",target_price)
print("매도가 조회 :",predicted_sell_price)
print("현재가 조회 :",current_price)
print("원화잔고 :",krw)
print("비트코인잔고 :",get_balance(COIN))
print("변동성 :",vola_break_price)
print("autotrade start")
# 스케줄러 실행
while True:
    try:
        schedule.run_pending()
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
