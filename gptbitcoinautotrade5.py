import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import schedule

access = "9PofI7vsCCOJEaSaxzZnW79HxcWHnQA2FbrQ7cWQ"
secret = "diUxCv8gLAl2QQ6Q0RT620as3Vxaon4vYrqyxjMc"
buy_unit = 0.25   # 분할 매수 금액 단위 설정
sell_unit = 0.25  # 분할 매도 금액 단위 설정

stop_loss = 0.95      # 손절률 5%
bought = False
sell_time = None
buy_price = None

COIN = "KRW-BTC" #코인명
days = 3 # 시작은 최근 3일 동안의 데이터로 설정

def get_target_price(ticker, k, days):
    # 최근 3+n일 동안의 데이터를 가져와서 매수 목표가 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=days+1)
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
    # 3일 동안의 데이터를 가져와서 매도 예측 가격 계산
    df = pyupbit.get_ohlcv(ticker, interval="3", count=days)
    ts = df['close']
    model = sm.tsa.ARIMA(ts, order=(2, 1, 2))
    results = model.fit(trend='nc', full_output=True, disp=1)
    forecast = results.forecast(steps=1)
    return forecast[0][0]

# 로그인
upbit = pyupbit.Upbit(access, secret)

# 자동매매 시작 함수

def run_auto_trade():
    global predicted_sell_price
    while True:
        try:
            now = datetime.datetime.now()
            start_time = get_start_time(COIN)
            end_time = start_time + datetime.timedelta(days=1)
            if start_time < now < end_time - datetime.timedelta(seconds=10):
                target_price = get_target_price(COIN, 0.7, days)
                current_price = get_current_price(COIN)
                if target_price < current_price:
                    krw = get_balance("KRW")
                    if krw > 5000:
                        buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
                        upbit.buy_market_order(COIN, buy_amount)
                        days += 1  # 분할 매수할 때마다 n일 증가
                        if days >= 7:
                            days = 3
            else:
                if predicted_sell_price is None or now.hour == 9 and now.minute == 0:
                    predicted_sell_price = predict_sell_price(COIN)
                current_price = get_current_price(COIN)
                if current_price >= predicted_sell_price:
                    btc = get_balance("BTC")
                    if btc > 0.00008:
                        sell_amount = btc * 1 * sell_unit # 분할 매도 금액 계산
                        upbit.sell_market_order(COIN, sell_amount)
                        predicted_sell_price = max(predicted_sell_price, predict_sell_price(COIN))
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
