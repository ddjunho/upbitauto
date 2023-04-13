import time
import datetime
import json
import pyupbit
import pandas as pd
import numpy as np
import schedule
import telepot
import tensorflow as tf
import requests.exceptions
import simplejson.errors
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
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
    df1 = pyupbit.get_ohlcv(ticker, interval="minute180", count=200)
    df2 = pyupbit.get_ohlcv(ticker, interval="minute180", count=200, to=df1.index[0])
    df3 = pyupbit.get_ohlcv(ticker, interval="minute180", count=200, to=df2.index[0])
    df4 = pyupbit.get_ohlcv(ticker, interval="minute180", count=200, to=df3.index[0])
    DF = pd.concat([df4, df3, df2, df1])
    # 입력 데이터 전처리
    X = DF[['open', 'high', 'low', 'close', 'volume']].values
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    # 출력 데이터 전처리
    y = DF[target_type].values
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.reshape((-1, 1)))
    # 학습 데이터 생성
    X_train = []
    y_train = []
    data=799
    for i in range(data, len(X)):
        X_train.append(X[i - data:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(data, 5)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    # 학습
    model.fit(X_train, y_train, epochs=100, verbose=1)
    # 새로운 데이터에 대한 예측
    last_data = DF[['open', 'high', 'low', 'close', 'volume']].iloc[-data:].values
    last_data_mean = last_data.mean(axis=0)
    last_data_std = last_data.std(axis=0)
    last_data = (last_data - last_data_mean) / last_data_std
    # 예측할 데이터의 shape를 (1,799, 5)로 변경
    last_data = np.expand_dims(last_data, axis=0)
    predicted_price = model.predict(last_data)
    predicted_price = y_scaler.inverse_transform(predicted_price)
    predicted_price = predicted_price.flatten()[0]  # 이중 리스트를 일차원으로 변경하고 첫 번째 원소를 선택
    return float(predicted_price)

def is_bull_market(ticker):
    global proba_3h
    global proba_6h 
    df1 = pyupbit.get_ohlcv(ticker, interval="minute10", count=200)
    df2 = pyupbit.get_ohlcv(ticker, interval="minute10", count=200, to=df1.index[0])
    df3 = pyupbit.get_ohlcv(ticker, interval="minute10", count=200, to=df2.index[0])
    df4 = pyupbit.get_ohlcv(ticker, interval="minute10", count=200, to=df3.index[0])
    DF = pd.concat([df4, df3, df2, df1])
    # 기술적 지표 추가
    DF['ma5'] = DF['close'].rolling(window=5).mean()
    DF['ma10'] = DF['close'].rolling(window=10).mean()
    DF['ma20'] = DF['close'].rolling(window=20).mean()
    DF['ma60'] = DF['close'].rolling(window=60).mean()
    DF['ma120'] = DF['close'].rolling(window=120).mean()
    # RSI 계산
    delta = DF['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    DF['rsi'] = 100 - (100 / (1 + rs))
    # MACD 계산
    exp1 = DF['close'].ewm(span=12, adjust=False).mean()
    exp2 = DF['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    DF['macd'] = macd
    DF['macdsignal'] = signal
    DF['macdhist'] = hist
    # 결측값 제거
    DF = DF.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'rsi', 'macd', 'macdsignal', 'macdhist']]
    y_3h = (DF['close'].shift(-18) > DF['close']).astype(int) # 3시간 뒤의 상승장 예측
    y_6h = (DF['close'].shift(-36) > DF['close']).astype(int) # 6시간 뒤의 상승장 예측
    # 학습 데이터와 검증 데이터 분리
    X_train, X_test, y_train_3h, y_test_3h = train_test_split(X, y_3h, test_size=0.2, shuffle=False)
    _, _, y_train_6h, y_test_6h = train_test_split(X, y_6h, test_size=0.2, shuffle=False)
    # 모델 구성
    model_3h = RandomForestClassifier(n_estimators=100, max_depth=10)
    model_6h = RandomForestClassifier(n_estimators=100, max_depth=10)
    # 학습
    model_3h.fit(X_train, y_train_3h)
    model_6h.fit(X_train, y_train_6h)
    # 예측 확률 계산
    proba_3h = model_3h.predict_proba(X_test.iloc[-1].values.reshape(1,-1))[0][1]
    proba_6h = model_6h.predict_proba(X_test.iloc[-1].values.reshape(1,-1))[0][1]
    proba_3h = round(proba_3h, 2)
    proba_6h = round(proba_6h, 2)
    if proba_3h >= 0.45 and proba_6h >= 0.45:
        return False
    else:
        return False
# 로그인
upbit = pyupbit.Upbit(access, secret)
krw = get_balance("KRW")
target_price = predict_target_price("low")
sell_price = predict_target_price("high")
current_price = get_current_price(COIN)
btc = get_balance("BTC")
PriceEase=round((sell_price-target_price)*0.1, 1)
multiplier = 1
last_buy_time = None
time_since_last_buy = None
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
bull_market = is_bull_market(COIN)
def send_message(message):
    bot = telepot.Bot(token="6296102104:AAFC4ddbh7gSgkGOdysFqEBUkIoWXw0-g5A")
    chat_id = "5820794752"
    bot.sendMessage(chat_id, message)
message = f"매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n현재가 조회 : {current_price}\n3시간뒤 상승 예측 : {proba_3h*100}%\n6시간뒤 상승 예측 : {proba_6h*100}%{bull_market}\n원화잔고 : {krw}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}"
send_message(message)
print("autotrade start")
# 스케줄러 실행
while True:
    try:
        now = datetime.now()
        current_price = get_current_price(COIN)
        if now.hour % 3 == 0 and now.minute == 0:
            if krw <= get_balance("KRW"):
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
            target_price = predict_target_price("low")
            sell_price = predict_target_price("high")
            PriceEase = round((sell_price - target_price) * 0.1, 1)
            bull_market = is_bull_market(COIN)
            message = f"매수가 조회 : {target_price}\n매도가 조회 : {sell_price}\n현재가 조회 : {current_price}\n3시간뒤 상승 예측 : {proba_3h*100}%\n6시간뒤 상승 예측 : {proba_6h*100}%{bull_market}\n원화잔고 : {krw}\n비트코인잔고 : {btc}\n목표가 완화 : {PriceEase}"
            send_message(message)
        # 매수 조건
        if current_price <= target_price + PriceEase:
            krw = get_balance("KRW")
            if bull_market==True and krw > 10000 and target_price + PriceEase < sell_price-(PriceEase*3):
                if get_balance("KRW") < krw * buy_unit:
                    buy_amount = krw * 0.9995
                upbit.buy_market_order(COIN, buy_amount)
                last_buy_time = datetime.now()
                multiplier = 1
                print(now, "매수")
        # 매도 조건
        else:
            if current_price >= sell_price-(PriceEase*multiplier):
                btc = get_balance("BTC")
                if btc > 0.00008 and btc is not None:
                    upbit.sell_market_order(COIN, btc)
                    print(now, "매도")
        # PriceEase 증가 조건
        if last_buy_time is not None:
            time_since_last_buy = now - last_buy_time
            if time_since_last_buy.total_seconds() >= 3600: # 1시간마다
                multiplier += 1
                if multiplier>3:
                    multiplier=4
                    last_buy_time = None
                last_buy_time = now
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
