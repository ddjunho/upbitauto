import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import schedule

access = "-"
secret = "-"
buy_unit = 0.1   # 분할 매수 금액 단위 설정
k = 0.35
COIN = "KRW-BTC" #코인명

def get_target_price(ticker):
    # 30일 동안의 데이터를 가져와서 매수 예측 가격 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
    ds = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    # ARIMA 모델 적용
    model = sm.tsa.arima.ARIMA(df['low'], order=(2, 1, 2))
    results = model.fit(method='statespace')
    forecast = results.forecast(steps=1).item()
    return forecast + (ds.iloc[0]['high'] - ds.iloc[0]['low']) * k

def stop_loss(ticker):
    # 2일 동안의 데이터를 가져와서 손절각보기
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    # ARIMA 모델 적용
    model = sm.tsa.arima.ARIMA(df['low'], order=(2, 1, 2))
    results = model.fit(method='statespace')
    forecast = results.forecast(steps=1).item()
    return forecast

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
    # 30일 동안의 데이터를 가져와서 매도 예측 가격 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=30)
    ds = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    # ARIMA 모델 적용
    model = sm.tsa.arima.ARIMA(df['high'], order=(2, 1, 2))
    results = model.fit(method='statespace')
    forecast = results.forecast(steps=1).item()
    return forecast - (ds.iloc[0]['high'] - ds.iloc[0]['low']) * k
# 로그인
upbit = pyupbit.Upbit(access, secret)

# 자동매매 시작 함수
krw = get_balance("KRW")
buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
target_price = get_target_price(COIN)
predicted_sell_price = predict_sell_price(COIN)
current_price = get_current_price(COIN)
stop_loss = stop_loss(COIN)
def run_auto_trade():
    global predicted_sell_price
    while True:
        try:
            now = datetime.datetime.now()
            if now.hour == 9 and now.minute == 0:
                krw = get_balance("KRW")
                buy_amount = krw * 0.9995 * buy_unit
                target_price = get_target_price(COIN)
                predicted_sell_price = predict_sell_price(COIN)
                current_price = get_current_price(COIN)
                stop_loss = stop_loss(COIN)
            if target_price >= current_price and target_price < predicted_sell_price and stop_loss < current_price:
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
