import os
import time
import telebot
import requests
from dotenv import load_dotenv
from config import get_company_settings

load_dotenv()

COMPANY_NAME = os.getenv("COMPANY_NAME", "defaultCompany")
settings = get_company_settings(COMPANY_NAME)

AI_MODEL = settings["AI_MODEL"]
BOT_TOKEN = settings["TELEGRAM_BOT_TOKEN"]
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000") + f"/chatbot/{COMPANY_NAME}"

bot = telebot.TeleBot(BOT_TOKEN)

bot_info = bot.get_me()
BOT_USERNAME = bot_info.username
BOT_NICKNAMES = [f"@{BOT_USERNAME}", "@AI봇", "@챗봇"]


@bot.message_handler(func=lambda message: (
    (message.text and any(tag in message.text for tag in BOT_NICKNAMES)) or
    (message.reply_to_message and message.reply_to_message.from_user.id == bot_info.id)
))
def handle_message(message):
    user_text = message.text or ""
    for tag in BOT_NICKNAMES:
        user_text = user_text.replace(tag, "").strip()

    chat_id = message.chat.id

    try:
        response = requests.post(BACKEND_URL, json={"message": user_text}, timeout=5)
        response.raise_for_status()
        bot_response = response.json().get("reply", "응답을 받지 못했습니다.")
    except requests.exceptions.HTTPError:
        bot_response = f"서버 오류 발생 (상태 코드: {response.status_code})"
    except requests.exceptions.RequestException as e:
        bot_response = f"서버 연결 오류 발생: {str(e)}"

    bot.send_message(chat_id, bot_response)


if __name__ == "__main__":
    while True:
        try:
            print(f"{COMPANY_NAME} 텔레그램 봇 실행 중...")
            bot.polling()

        except Exception as e:
            print(f"봇 오류 발생: {str(e)}")
            time.sleep(5)  # 5초 대기 후 재시작
