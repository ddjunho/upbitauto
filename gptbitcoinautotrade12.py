import time
import json
import asyncio
import pyupbit
import pandas as pd
import numpy as np
import schedule
import tensorflow as tf
import requests.exceptions
import simplejson.errors
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from fbprophet import Prophet
from upbit_keys import access, secret
from telegram import Bot
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
    df1 = pyupbit.get_ohlcv(ticker, interval="minute360", count=200)
    df2 = pyupbit.get_ohlcv(ticker, interval="minute360", count=200, to=df1.index[0])
    df3 = pyupbit.get_ohlcv(ticker, interval="minute360", count=150, to=df2.index[0])
    DF = pd.concat([df3, df2, df1])
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
    for i in range(549, len(X)):
        X_train.append(X[i - 549:i, :])
        y_train.append(y[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    # Tensorflow 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(549, 5)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mse', run_eagerly=True)
    # 학습
    model.fit(X_train, y_train, epochs=200, verbose=1)
    # 새로운 데이터에 대한 예측
    last_data = DF[['open', 'high', 'low', 'close', 'volume']].iloc[-549:].values
    last_data_mean = last_data.mean(axis=0)
    last_data_std = last_data.std(axis=0)
    last_data = (last_data - last_data_mean) / last_data_std
    # 예측할 데이터의 shape를 (1,549, 5)로 변경
    last_data = np.expand_dims(last_data, axis=0)
    predicted_price = model.predict(last_data)
    predicted_price = y_scaler.inverse_transform(predicted_price)
    predicted_price = predicted_price.flatten()[0]  # 이중 리스트를 일차원으로 변경하고 첫 번째 원소를 선택
    return float(predicted_price)

def is_bull_market(ticker):
    global proba
    df1 = pyupbit.get_ohlcv(ticker, interval="day", count=183)
    df2 = pyupbit.get_ohlcv(ticker, interval="day", count=183, to=df1.index[0])
    DF = pd.concat([df2, df1])
    # 기술적 지표 추가
    DF['ma5'] = DF['close'].rolling(window=5).mean()
    DF['ma10'] = DF['close'].rolling(window=10).mean()
    DF['ma20'] = DF['close'].rolling(window=20).mean()
    DF['ma60'] = DF['close'].rolling(window=60).mean()
    DF['ma120'] = DF['close'].rolling(window=120).mean()
    # 결측값 제거
    DF = DF.dropna()
    # 입력 데이터와 출력 데이터 분리
    X = DF[['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 'ma60', 'ma120']]
    y = (DF['close'].shift(-1) > DF['close']).astype(int)
    # 학습 데이터와 검증 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # 모델 구성
    model = RandomForestClassifier(n_estimators=100)
    # 학습
    model.fit(X_train, y_train)
    # 예측 확률 계산
    proba = model.predict_proba(X_test.iloc[-1].values.reshape(1,-1))[0][1]
    # 조건 검사
    if proba >= 0.45:
        return True
    else:
        return False

close_price = 0
def predict_price(ticker):
    global close_price
    df = pyupbit.get_ohlcv(ticker, interval="minute60")
    df = df.reset_index()
    df['ds'] = df['index']
    df['y'] = df['close']
    data = df[['ds','y']]
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    # 3시, 9시, 15시, 21시 종가 예측
    close_values = []
    for hour in [3, 9, 15, 21]:
        close_df = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=hour)]
        if len(close_df) == 0:
            close_df = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=hour)]
        close_value = close_df['yhat'].values[0]
        close_values.append(close_value)
    # 결과 저장
    close_price = tuple(close_values)
predict_price("KRW-BTC")
schedule.every().hour.do(lambda: predict_price("KRW-BTC"))


# 로그인
upbit = pyupbit.Upbit(access, secret)
krw = get_balance("KRW")
target_price = predict_target_price("low")
sell_price = predict_target_price("high")
current_price = get_current_price(COIN)
btc = get_balance("BTC")
bull_market = is_bull_market(COIN)
PriceEase=round((sell_price-target_price)*0.1, 1)
multiplier = 1
last_buy_time = None
time_since_last_buy = None
is_tradeable = False
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
async def chat_bot():
    # proba 값을 Telegram으로 전송
    bot_token = "5915962696:AAF14G7Kg-N2tk5i_w4JGYICqamwrUNXP1c" # 봇 토큰
    bot_chat_id = "5820794752" # 채팅 ID
    bot = Bot(token=bot_token)
    message = "매수가 조회 : {}\n매도가 조회 : {}\n현재가 조회 : {}\n상승장 예측 : {} {}\n원화잔고 : {}\n비트코인잔고 : {}\n목표가 완화 : {}".format(target_price, sell_price, current_price, proba, bull_market, krw, btc, PriceEase*3)
    await bot.send_message(chat_id=bot_chat_id, text=message)
    if bull_market==True:
        message = "45%이상으로 예측되므로 매매를 시작합니다. \n\n★Autotrade start★"
        await bot.send_message(chat_id=bot_chat_id, text=message)
asyncio.run(chat_bot())
chat_bot()
schedule.every().day.at("09:00").do(chat_bot)
print("autotrade start")
# 스케줄러 실행
while True:
    try:
        schedule.run_pending()
        now = datetime.now()
        current_price = get_current_price(COIN)
        if now.hour in [3, 9, 15, 21] and now.minute == 0:
            if krw <= get_balance("KRW"):
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
            target_price = predict_target_price(COIN, 'low')
            sell_price = predict_target_price(COIN, 'high')
            PriceEase = round((sell_price - target_price) * 0.1, 1)
            bull_market = is_bull_market(COIN)
        if now.hour < 3:
            if current_price < close_price[0]:
                is_tradeable = True
            else:
                is_tradeable = False
        elif 3 <= now.hour < 9:
            if current_price < close_price[1]:
                is_tradeable = True
            else:
                is_tradeable = False
        elif 9 <= now.hour < 15: 
            if current_price < close_price[2]:
                is_tradeable = True
            else:
                is_tradeable = False
        elif 15 <= now.hour < 21:
            if current_price < close_price[3]:
                is_tradeable = True
            else:
                is_tradeable = False
        # 매수 조건
        if current_price <= target_price + PriceEase*2:
            if bull_market==True and is_tradeable == True and krw > 10000 and target_price + PriceEase*2 < sell_price-(PriceEase*3):
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
