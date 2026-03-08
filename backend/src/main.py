from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. CORSの設定（ローカルテスト用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# 2. 新しいGemini APIクライアントの初期化
client = genai.Client()

# システムインストラクションを読み込む
SYSTEM_INSTRUCTION = ""
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "models.txt")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except Exception as e:
    print(f"Warning: Failed to load system instruction from {config_path}: {e}")

# データの受け取りフォーマット
class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-flash-latest"

# 3. エンドポイントの作成
@app.post("/api/chat")
def chat_with_gemini(req: ChatRequest):
    try:
        # 新しいSDKの書き方でGeminiにリクエストを送信
        response = client.models.generate_content(
            model=req.model,
            contents=req.message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )
        return {"reply": response.text}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/models")
def list_models():
    try:
        # Gemini APIから使えるモデルの一覧を取得
        models = client.models.list()
        # モデルの名前（name属性）だけを抽出してリストにして返す
        return {"available_models": [model.name for model in models]}
    except Exception as e:
        return {"error": str(e)}