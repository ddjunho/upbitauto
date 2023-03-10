import time
import pyupbit
import datetime
import schedule
from fbprophet import Prophet

access = "9PofI7vsCCOJEaSaxzZnW79HxcWHnQA2FbrQ7cWQ"
secret = "diUxCv8gLAl2QQ6Q0RT620as3Vxaon4vYrqyxjMc"

def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

predicted_sell_price = 0

def predict_price(ticker):
    """Prophet으로 매도 예측 가격 예측"""
    global predicted_sell_price
    df = pyupbit.get_ohlcv(ticker, interval="minute60")
    df = df.reset_index()
    df['ds'] = df['index']
    df['y'] = df['close']
    data = df[['ds','y']]
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    sellDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=23, minute=50)]
    if len(sellDf) == 0:
        sellDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=23, minute=50)]
    sellValue = sellDf['yhat'].values[0]
    predicted_sell_price = sellValue

predict_price("KRW-BTC")
schedule.every().hour.do(lambda: predict_price("KRW-BTC"))

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-BTC")
        end_time = start_time + datetime.timedelta(days=1)
        schedule.run_pending()
        if start_time < now < end_time - datetime.timedelta(seconds=10):
            target_price = get_target_price("KRW-BTC", 0.7)
            current_price = get_current_price("KRW-BTC")
            if target_price < current_price and current_price < predicted_close_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    upbit.buy_market_order("KRW-BTC", krw*0.9995)
        else:
            if now.hour == 23 and now.minute >= 50:
                btc = get_balance("BTC")
                if btc > 0.00008:
                    upbit.sell_market_order("KRW-BTC", btc*1)
            else:
                predict_price("KRW-BTC")
        time.sleep(1)
    except Exception as e:
        print(e)
        time.sleep(1)
