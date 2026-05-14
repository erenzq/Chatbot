import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionAssistantMessageParam
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_embeddings = []
log_texts = []


class ChatRequest(BaseModel):
    question: str
    history: List[dict]


def initialize_embeddings(filepath):
    global log_embeddings, log_texts

    log_embeddings = []
    log_texts = []

    with open(filepath, "r") as file:
        for line in file:
            log_text = line.strip()
            embedding = get_embedding(log_text)

            log_texts.append(log_text)
            log_embeddings.append(embedding)


def get_embedding(text):
    response = client.embeddings.create(
        model="Qwen/Qwen3-Embedding-8B",
        input=text
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_relevant_logs_embeddings(question, top_k=3):
    question_embedding = get_embedding(question)

    scored_logs = []

    for text, embedding in zip(log_texts, log_embeddings):
        score = cosine_similarity(question_embedding, embedding)
        scored_logs.append((text, score))

    scored_logs.sort(key=lambda x: x[1], reverse=True)
    top_logs = [log for log, _ in scored_logs[:top_k]]

    formatted = ""

    for i, log in enumerate(top_logs, 1):
        formatted += f"{i}. {log}\n"

    return formatted


@app.post("/ask_ai_stream")
def ask_ai_stream(request: ChatRequest):
    context = get_relevant_logs_embeddings(request.question)
    system_msg: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": (
            "Du bist ein Experte für Fahrzeug-Logs.\n"
            "Analysiere Logs präzise und faktenbasiert.\n"
            "Nutze nur die gegebenen Informationen.\n\n"
            "Antworte strukturiert:\n"
            "Problem:\n"
            "Ursache:\n"
            "Lösung:\n"
        )
    }
    max_history = 5
    history_messages = []

    for entry in request.history:
        if entry["role"] == "user":
            history_messages.append(ChatCompletionUserMessageParam(
                role="user",
                content=entry["content"]
            ))
        elif entry["role"] == "assistant":
            history_messages.append(ChatCompletionAssistantMessageParam(
                role="assistant",
                content=entry["content"]
            ))
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": (
            "Hier sind relevante Logs:\n"
            f"{context}\n\n"
            "Beantworte folgende Frage basierend darauf:\n"
            f"{request.question}"
        )
    }
    messages = [system_msg] + history_messages[-max_history:] + [user_msg]

    def generate():
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/upload_logs")
def upload_logs(file: UploadFile = File(...)):
    global log_embeddings, log_texts

    log_embeddings = []
    log_texts = []

    content = file.file.read().decode("utf-8")
    lines = content.splitlines()

    for line in lines:
        log_text = line.strip()
        if log_text:
            embedding = get_embedding(log_text)
            log_texts.append(log_text)
            log_embeddings.append(embedding)

    return {"message": f"Successfully uploaded and processed {len(log_texts)} logs."}


initialize_embeddings("log.txt")
