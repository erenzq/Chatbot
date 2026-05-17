import os
from collections import Counter
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_embeddings = []
log_texts = []
parsed_logs = []
embedding_cache = {}


class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []


# -------------------------
# EMBEDDINGS
# -------------------------

def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    response = client.embeddings.create(
        model="Qwen/Qwen3-Embedding-8B",
        input=text
    )
    embedding = np.array(response.data[0].embedding)
    embedding_cache[text] = embedding
    return embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_relevant_logs_embeddings(question, top_k=3):
    if not log_embeddings:
        return "Keine Logs geladen."

    question_embedding = get_embedding(question)

    scored = [
        (text, cosine_similarity(question_embedding, emb))
        for text, emb in zip(log_texts, log_embeddings)
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    return "\n".join(f"{i + 1}. {log}" for i, (log, _) in enumerate(scored[:top_k]))


# -------------------------
# LOG ANALYSIS
# -------------------------

def parse_log(line):
    parts = line.split(":", 1)
    if len(parts) == 2:
        return {"level": parts[0].strip(), "message": parts[1].strip()}
    return {"level": "UNKNOWN", "message": line.strip()}


def get_log_stats():
    if not parsed_logs:
        return {}
    return dict(Counter(log["level"] for log in parsed_logs))


def get_root_causes(top_k=3):
    msgs = [log["message"] for log in parsed_logs]
    return Counter(msgs).most_common(top_k)


def detect_anomalies(threshold=1):
    msgs = [log["message"] for log in parsed_logs]
    counter = Counter(msgs)
    return [msg for msg, c in counter.items() if c <= threshold]


def compute_confidence():
    if not parsed_logs:
        return 0.0
    return round(min(1.0, len(parsed_logs) / 50), 2)


# -------------------------
# MESSAGE BUILDER
# -------------------------

def build_messages(question, history):
    context = get_relevant_logs_embeddings(question)
    stats = get_log_stats()
    root = get_root_causes()
    anomalies = detect_anomalies()
    confidence = compute_confidence()

    # 🔥 sichere history
    safe_history = []
    for entry in history[-5:]:
        role = entry.get("role")
        content = entry.get("content")

        if role in ["user", "assistant"] and content:
            safe_history.append({
                "role": role,
                "content": content
            })

    system_msg = {
        "role": "system",
        "content": (
            "Du bist ein Experte für Fahrzeug-Logs.\n"
            "Antworte strukturiert:\n"
            "Problem:\nUrsache:\nLösung:\nConfidence:\n"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Statistiken:\n{stats}\n\n"
            f"Häufigste Fehler:\n{root}\n\n"
            f"Auffällige Logs:\n{anomalies}\n\n"
            f"Confidence: {confidence}\n\n"
            f"Logs:\n{context}\n\n"
            f"Frage:\n{question}\n"
        )
    }

    return [system_msg] + safe_history + [user_msg]


# -------------------------
# ROUTES
# -------------------------

@app.post("/ask_ai_stream")
def ask_ai_stream(request: ChatRequest):
    messages = build_messages(request.question, request.history)

    def generate():
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/ask_ai")
def ask_ai(request: ChatRequest):
    messages = build_messages(request.question, request.history)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )

    return {"answer": response.choices[0].message.content}


@app.post("/upload_logs")
def upload_logs(file: UploadFile = File(...)):
    global log_embeddings, log_texts, parsed_logs

    log_embeddings.clear()
    log_texts.clear()
    parsed_logs.clear()

    lines = file.file.read().decode("utf-8").splitlines()

    for line in lines:
        if not line.strip():
            continue

        parsed = parse_log(line)
        parsed_logs.append(parsed)

        emb = get_embedding(line)
        log_embeddings.append(emb)
        log_texts.append(line)

    return {"message": f"{len(log_texts)} logs geladen."}


@app.get("/")
def root():
    return FileResponse("static/index.html")


# -------------------------
# START
# -------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
