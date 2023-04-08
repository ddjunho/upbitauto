import asyncio
from telegram import Bot

async def send_message(bot_token, bot_chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=bot_chat_id, text=message)

bot_token = "5915962696:AAF14G7Kg-N2tk5i_w4JGYICqamwrUNXP1c" # 봇 토큰
bot_chat_id = "5820794752" # 채팅 ID
message = "Hello, World!"

asyncio.run(send_message(bot_token, bot_chat_id, message))
