import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import schedule

access = "9PofI7vsCCOJEaSaxzZnW79HxcWHnQA2FbrQ7cWQ"
secret = "diUxCv8gLAl2QQ6Q0RT620as3Vxaon4vYrqyxjMc"
buy_unit = 0.1   # 분할 매수 금액 단위 설정
sell_unit = 0.1  # 분할 매도 금액 단위 설정

target_profit = 1.05  # 목표 수익률 5%
stop_loss = 0.95      # 손절률 5%

bought = False
sell_time = None
buy_price = None

COIN = "KRW-BTC" #코인명

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

def predict_sell_price(ticker):
    # 일주일 동안의 데이터를 가져와서 매도 예측 가격 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=7)
    ts = df['close']
    model = sm.tsa.ARIMA(ts, order=(2, 1, 2))
    results = model.fit(trend='nc', full_output=True, disp=1)
    forecast = results.forecast(steps=1)
    return forecast[0][0]

# 로그인
upbit = pyupbit.Upbit(access, secret)

# 자동매매 시작 함수
def run_auto_trade():
    predicted_sell_price = predict_sell_price(COIN)
    while True:
        try:
            now = datetime.datetime.now()
            start_time = get_start_time(COIN)
            end_time = start_time + datetime.timedelta(days=1)
            if start_time < now < end_time - datetime.timedelta(seconds=10):
                target_price = get_target_price(COIN, 0.7)
                current_price = get_current_price(COIN)
                if target_price < current_price:
                    krw = get_balance("KRW")
                    if krw > 5000:
                        upbit.buy_market_order(COIN, krw*0.9995)
            else:
                current_price = get_current_price(COIN)
                if current_price >= predicted_sell_price:
                    btc = get_balance("BTC")
                    if btc > 0.00008:
                        upbit.sell_market_order(COIN, btc*1)
                        predicted_sell_price = predict_sell_price("KRW-BTC")
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(1)
