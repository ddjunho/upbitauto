import pyupbit
import numpy as np

#ohlcv=시가(open), 고가(high), 저가(low), 종가(close), 거래량(volume) 7일간데이터
df = pyupbit.get_ohlcv("KRW-BTC",count=14)
#전날변동폭(range) 계산
df['range'] = (df['high'] - df['low']) * 0.7
#메수가(target)
df['target'] = df['open'] + df['range'].shift(1)
#수익률(ror) 조건문(np.where)
df['ror'] = np.where(df['high'] > df['target'],
                     df['close'] / df['target'],
                     1)
#수익률누적(cumprod)(hpr)
df['hpr'] = df['ror'].cumprod()
#하락폭(Draw Down)(dd)
df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100
#최대하락폭
print("MDD(%): ", df['dd'].max())
#엑셀로 출력
df.to_excel("dd.xlsx")