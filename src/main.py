import os
import base64
import tempfile
import json
import re
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY         = os.environ.get("X_API_KEY", "sk_track3_987654321")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "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")
WHISPER_MODEL   = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
LLM_MODEL       = os.environ.get("LLM_MODEL", "meta-llama/llama-3-70b-instruct")

# ---------------------------------------------------------------------------
# Whisper Load
# ---------------------------------------------------------------------------
app = FastAPI()

# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------
ALLOWED_LANGUAGES = {"Tamil", "Hindi"}

class CallRequest(BaseModel):
    language: str
    audioBase64: str

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if v.capitalize() not in ALLOWED_LANGUAGES:
            raise ValueError("Language must be Tamil or Hindi")
        return v.capitalize()

# ---------------------------------------------------------------------------
# Decode Audio
# ---------------------------------------------------------------------------
def decode_audio(audio_b64: str) -> str:
    audio_bytes = base64.b64decode(audio_b64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name

# ---------------------------------------------------------------------------
# Whisper STT
# ---------------------------------------------------------------------------
LANGUAGE_CODE = {"Tamil": "ta", "Hindi": "hi"}

def transcribe(audio_path: str, language: str) -> str:
    client = OpenAI(
        api_key=os.getenv("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"),
        base_url="https://api.groq.com/openai/v1"
    )

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3"
        )

    return response.text.strip()


# ---------------------------------------------------------------------------
# Detect Native Script
# ---------------------------------------------------------------------------
def contains_tamil(text):
    return re.search(r'[\u0B80-\u0BFF]', text)

def contains_hindi(text):
    return re.search(r'[\u0900-\u097F]', text)

# ---------------------------------------------------------------------------
# 🔥 Convert to Tanglish / Hinglish
# ---------------------------------------------------------------------------
def convert_to_roman(text: str, language: str) -> str:

    # If already roman → skip
    if not contains_tamil(text) and not contains_hindi(text):
        return text

    try:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

        prompt = f"""
Convert this text into natural spoken {"Tanglish" if language=="Tamil" else "Hinglish"}.

Rules:
- Keep English words unchanged
- Convert Tamil/Hindi words to Roman script
- Do NOT translate meaning
- Output only the sentence

Text:
{text}
"""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a transliteration expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        log.warning("Roman conversion failed: %s", e)
        return text

# ---------------------------------------------------------------------------
# LLM Analysis
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "Return JSON with summary, sop_validation, analytics, keywords."

def analyze(transcript: str) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------------
@app.post("/api/call-analytics")
async def call_analytics(req: CallRequest, x_api_key: str = Header(...)):

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_path = None

    try:
        audio_path = decode_audio(req.audioBase64)

        # Step 1: STT
        transcript = transcribe(audio_path, req.language)

        # 🔥 Step 2: Convert to Tanglish/Hinglish
        transcript = convert_to_roman(transcript, req.language)

        # Step 3: Analysis
        analysis = analyze(transcript)

        return {
            "status": "success",
            "language": req.language,
            "transcript": transcript,
            **analysis
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
