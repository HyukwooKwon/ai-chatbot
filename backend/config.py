import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")

COMPANY_NAMES = os.getenv("COMPANY_NAMES", "").split(",")

COMPANY_AI_MODELS = {name: os.getenv(f"AI_MODEL_{name}", "gpt-3.5-turbo") for name in COMPANY_NAMES}

API_KEYS = {
    model: os.getenv(f"OPENAI_API_KEY_{model}") for model in set(COMPANY_AI_MODELS.values())
}

ALLOWED_COMPANIES = set(COMPANY_NAMES)


def get_company_settings(company_name):
    if company_name not in ALLOWED_COMPANIES:
        raise ValueError(f"지원되지 않는 업체입니다: {company_name}")

    ai_model = COMPANY_AI_MODELS[company_name]
    openai_api_key = API_KEYS.get(ai_model)

    telegram_bot_token = os.getenv(f"TELEGRAM_BOT_TOKEN_{ai_model}")
    telegram_upload_bot_token = os.getenv("TELEGRAM_BOT_TOKEN_UPLOAD")
    telegram_chat_id = os.getenv(f"TELEGRAM_CHAT_ID_{company_name}")

    missing_vars = [
        var_name for var_name, var_value in [
            ("AI_MODEL", ai_model),
            ("OPENAI_API_KEY", openai_api_key),
            ("TELEGRAM_BOT_TOKEN", telegram_bot_token),
            ("TELEGRAM_BOT_TOKEN_UPLOAD", telegram_upload_bot_token),
            ("TELEGRAM_CHAT_ID", telegram_chat_id)
        ] if not var_value
    ]

    if missing_vars:
        missing = ", ".join(missing_vars)
        raise ValueError(f"{company_name} 설정이 누락됨: {missing}")

    return {
        "AI_MODEL": ai_model,
        "OPENAI_API_KEY": openai_api_key,
        "TELEGRAM_BOT_TOKEN": telegram_bot_token,
        "TELEGRAM_BOT_TOKEN_UPLOAD": telegram_upload_bot_token,
        "TELEGRAM_CHAT_ID": telegram_chat_id
    }


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 영구 디스크 사용을 위한 경로 설정 (Render 등 배포환경용)
DATABASE_DIR = os.getenv("DATABASE_DIR", "./databases")
