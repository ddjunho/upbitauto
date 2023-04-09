# upbitauto

nohup python3 gptbitcoinautotrade.py > output.log 2>&1 &

백그라운드 실행시 표준 오류 출력을 표준 출력을 모두 지정된 파일로 리디렉션


gptbitcoinautotrade6.py -> 매도예측최고가에 매도하도록 설정된 코드

gptbitcoinautotrade11.py -> 6시간 단위로 3개월 모델학습

gptbitcoinautotrade12.py -> 매일 오전 9시에 실행 값을 Telegram으로 전송
gptbitcoinautotrade15.py -> 모델의 손실함수에 L2 규제를 적용하고 규제 강도를 0.01로 설정
