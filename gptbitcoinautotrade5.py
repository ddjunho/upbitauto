import time
import pyupbit
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import schedule

access = "-"
secret = "-"
buy_unit = 0.25   # 분할 매수 금액 단위 설정
sell_unit = 0.5  # 분할 매도 금액 단위 설정

stop_loss = 0.95      # 손절률 5%
bought = False
sell_time = None
buy_price = None
day_s = 0
COIN = "KRW-BTC" #코인명

def get_target_price(ticker, k):
    # 최근 3+n일 동안의 데이터를 가져와서 매수 목표가 계산
    global day_s
    day_s += 1  # 분할 매수할 때마다 n일 증가
    if day_s >= 3:
        day_s = 0
    df = pyupbit.get_ohlcv(ticker, interval="day", count=day_s+4)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price
  
def get_start_time(ticker):
    # 시작 시간 09:00
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0].replace(hour=9, minute=0, second=0, microsecond=0)
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
    # 7일 동안의 데이터를 가져와서 매도 예측 가격 계산
    df = pyupbit.get_ohlcv(ticker, interval="day", count=7)
    ts = df['close']
    model = sm.tsa.ARIMA(ts, order=(2, 1, 2))
    results = model.fit(trend='nc', full_output=True, disp=1)
    forecast = results.forecast(steps=1)
    return forecast[0][0]

# 로그인
upbit = pyupbit.Upbit(access, secret)

# 자동매매 시작 함수
predicted_sell_price=None
krw = get_balance("KRW")
def run_auto_trade():
    global predicted_sell_price
    while True:
        try:
            now = datetime.datetime.now()
            start_time = get_start_time(COIN)
            end_time = start_time + datetime.timedelta(days=1)
            if start_time < now < end_time - datetime.timedelta(seconds=10):
                target_price = get_target_price(COIN, 0.7)
                current_price = get_current_price(COIN)
                if target_price < current_price:
                    buy_amount = krw * 0.9995 * buy_unit # 분할 매수 금액 계산
                    if get_balance("KRW") < krw * buy_unit:
                        buy_amount = krw * 0.9995
                    upbit.buy_market_order(COIN, buy_amount)
            else:
                if predicted_sell_price is None or now.hour == 9 and now.minute == 0:
                    predicted_sell_price = predict_sell_price(COIN)
                current_price = get_current_price(COIN)
                if current_price >= predicted_sell_price:
                    btc = get_balance("BTC")
                    if btc > 0.00008:
                        sell_amount = btc * 1
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
