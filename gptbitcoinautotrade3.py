import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import schedule

# API Key 설정
access = "9PofI7vsCCOJEaSaxzZnW79HxcWHnQA2FbrQ7cWQ"
secret = "diUxCv8gLAl2QQ6Q0RT620as3Vxaon4vYrqyxjMc"

def get_target_price(ticker, k):
    # 최근 24시간 동안의 데이터를 가져와서 매수 목표가 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price
  
def get_start_time(ticker):
    # 최근 1일 동안의 데이터를 가져와서 시작 시간 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time
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
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]
  
# 매도 예측 함수
def predict_sell_price(ticker):
    df = pyupbit.get_ohlcv(ticker, interval="day", count=7)
    ts = df['close']
    model = sm.tsa.ARIMA(ts, order=(2, 1, 2))
    results = model.fit(trend='nc', full_output=True, disp=1)
    forecast = results.forecast(steps=1)
    return forecast[0][0]

# 로그인
upbit = pyupbit.Upbit(access, secret)

# 매도 예측 초기값 설정
predicted_sell_price = predict_sell_price("KRW-BTC")

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1)
        if start_time < now < end_time - datetime.timedelta(seconds=10):
            # 매수 조건 검사
            target_price = get_target_price("KRW-BTC", 0.7)
            current_price = get_current_price("KRW-BTC")
            if target_price < current_price and current_price < buy_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    upbit.buy_market_order("KRW-BTC", krw*0.9995)
        else:
            # 매도 조건 검사
            current_price = get_current_price("KRW-BTC")
            if current_price >= predicted_sell_price:
                btc = get_balance("BTC")
                if btc > 0.00008:
                    upbit.sell_market_order("KRW-BTC", btc*1)
                # 매도 후 다시 예측
                predicted_sell_price = predict_sell_price("KRW-BTC")
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)