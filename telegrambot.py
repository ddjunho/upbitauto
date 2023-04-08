from telegram import Bot

bot_token = "5915962696:AAF14G7Kg-N2tk5i_w4JGYICqamwrUNXP1c" # 봇 토큰
bot_chat_id = "5820794752" # 채팅 ID
bot = Bot(token=bot_token)

message = "Hello, World!"
bot.sendMessage(chat_id=bot_chat_id, text=message)
