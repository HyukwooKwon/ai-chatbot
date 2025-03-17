import os
import telebot
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from datetime import datetime
from chatbot import get_chatbot_response
from create_vector_db import create_or_update_faiss
from config import get_company_settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True)
    user_message = Column(String, nullable=False)
    bot_response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)


class Inquiry(Base):
    __tablename__ = "inquiries"
    id = Column(Integer, primary_key=True)
    contact = Column(String, nullable=False)
    inquiry = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)


class ChatInput(BaseModel):
    message: str


class InquiryInput(BaseModel):
    contact: str
    inquiry: str


def get_company_db(company_name):
    db_path = f"databases/{company_name}.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def send_telegram_notification(bot_token, chat_id, company_name, user_message, bot_response):
    try:
        telegram_bot = telebot.TeleBot(bot_token)
        telegram_bot.send_message(
            chat_id,
            f"📌 [{company_name}의 챗봇 기록]\n\n👤질문:\n{user_message}\n\n🤖답변:\n{bot_response}"
        )
    except Exception as e:
        print(f"텔레그램 메시지 전송 실패: {str(e)}")


@app.post("/chatbot/{company_name}")
def chatbot(company_name: str, chat: ChatInput):
    settings = get_company_settings(company_name)

    user_message = chat.message.strip()

    if any(keyword in user_message for keyword in ["그림", "이미지", "그려", "생성"]):
        try:
            client = OpenAI(api_key=settings["OPENAI_API_KEY"])
            response = client.images.generate(model="dall-e-3", prompt=user_message, size="1024x1024", n=1)
            bot_response = f"이미지를 생성했습니다: {response.data[0].url}"
        except Exception as e:
            bot_response = f"이미지 생성 실패: {str(e)}"
    else:
        bot_response = get_chatbot_response(
            user_message, company_name, settings["AI_MODEL"], settings["OPENAI_API_KEY"]
        )

    Session = get_company_db(company_name)
    with Session() as session:
        new_chat = ChatHistory(user_message=user_message, bot_response=bot_response)
        session.add(new_chat)
        session.commit()

    send_telegram_notification(
        settings["TELEGRAM_BOT_TOKEN_UPLOAD"],
        settings["TELEGRAM_CHAT_ID"],
        company_name,
        user_message,
        bot_response
    )

    return {"reply": bot_response}


@app.post("/chatbot/{company_name}/kakao")
async def kakao_chatbot(company_name: str, request: Request):
    body = await request.json()
    user_message = body["userRequest"]["utterance"].strip()

    settings = get_company_settings(company_name)
    bot_response = get_chatbot_response(user_message, company_name, "gpt-3.5-turbo", settings["OPENAI_API_KEY"])

    return {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": bot_response}}]}}


@app.get("/chatbot/history/{company_name}")
def get_chat_history(company_name: str, limit: int = 10):
    Session = get_company_db(company_name)
    with Session() as session:
        history = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
    return {"history": [{"message": h.user_message, "reply": h.bot_response, "timestamp": h.timestamp} for h in history]}


@app.post("/submit-inquiry/{company_name}")
def submit_inquiry(company_name: str, inquiry: InquiryInput):
    Session = get_company_db(company_name)
    with Session() as session:
        session.add(Inquiry(contact=inquiry.contact, inquiry=inquiry.inquiry))
        session.commit()
    return {"message": f"{company_name}의 문의가 저장되었습니다."}


@app.get("/inquiries/{company_name}")
def get_inquiries(company_name: str):
    Session = get_company_db(company_name)
    with Session() as session:
        inquiries = session.query(Inquiry).order_by(Inquiry.timestamp.desc()).all()
    return [{"contact": i.contact, "inquiry": i.inquiry, "timestamp": i.timestamp} for i in inquiries]


@app.post("/update-db/{company_name}")
def update_db(company_name: str):
    try:
        create_or_update_faiss(company_name)
        return {"message": f"{company_name}의 벡터DB가 업데이트되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
