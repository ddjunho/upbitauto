import pyupbit

access = "9PofI7vsCCOJEaSaxzZnW79HxcWHnQA2FbrQ7cWQ"          # 본인 값으로 변경
secret = "diUxCv8gLAl2QQ6Q0RT620as3Vxaon4vYrqyxjMc"          # 본인 값으로 변경
upbit = pyupbit.Upbit(access, secret)
   # KRW-BTC 조회
print(upbit.get_balance("KRW"))