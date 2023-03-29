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
from fbprophet import Prophet
from upbit_keys import access, secret
tf.config.run_functions_eagerly(True)
buy_unit = 0.2  # 분할 매수 금액 단위 설정

COIN = "KRW-BTC" #코인명

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
    predicted_price = predicted_price.flatten()[0]  # 이중 리스트를 일차원으로 변경하고 첫 번째 원소를 선택
    return float(predicted_price)

predicted_close_price = 0
def predict_price(ticker):
    """Prophet으로 당일 종가 가격 예측"""
    global predicted_close_price
    df = pyupbit.get_ohlcv(ticker, interval="minute60")
    df = df.reset_index()
    df['ds'] = df['index']
    df['y'] = df['close']
    data = df[['ds','y']]
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
    if len(closeDf) == 0:
        closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]
    closeValue = closeDf['yhat'].values[0]
    predicted_close_price = closeValue
predict_price("KRW-BTC")
schedule.every().hour.do(lambda: predict_price("KRW-BTC"))

# 로그인
upbit = pyupbit.Upbit(access, secret)
krw = get_balance("KRW")
target_price = predict_target_price("low")
predicted_sell_price = predict_target_price("high")
current_price = get_current_price(COIN)
btc = get_balance(COIN)
PriceEase=(predicted_sell_price-target_price)*0.15
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
print("매수가 조회 :",target_price)
print("매도가 조회 :",predicted_sell_price)
print("현재가 조회 :",current_price)
print("원화잔고 :",krw)
print("비트코인잔고 :",btc)
print("목표가 완화 :",PriceEase)
print("autotrade start")
# 스케줄러 실행
while True:
    try:
        schedule.run_pending()
        now = datetime.now()
        current_price = get_current_price(COIN)
        if now.hour == 9 and now.minute == 0 :
            if krw <= get_balance("KRW"):
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
            target_price = predict_target_price(COIN, 'low')
            predicted_sell_price = predict_target_price(COIN, 'high')
            PriceEase=(predicted_sell_price-target_price)*0.15
        if krw is not None and current_price <= target_price+PriceEase and target_price+PriceEase < predicted_sell_price-PriceEase and current_price < predicted_close_price:
            if krw > 10000:
                if get_balance("KRW") < krw * buy_unit:
                    buy_amount = krw * 0.9995
                upbit.buy_market_order(COIN, buy_amount)
                print(now, "매수")
        else:
            if current_price >= predicted_sell_price-PriceEase:
                btc = get_balance(COIN)
                if btc != 0 and btc is not None:
                    upbit.sell_market_order(COIN, btc)
                    print(now, "매도")
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
