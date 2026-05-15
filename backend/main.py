import os
from collections import Counter
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
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
parsed_logs = []


class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []


def get_embedding(text):
    response = client.embeddings.create(
        model="Qwen/Qwen3-Embedding-8B",
        input=text
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_relevant_logs_embeddings(question, top_k=3):
    if not log_embeddings:
        return "Keine Logs geladen."

    question_embedding = get_embedding(question)

    scored_logs = []

    for text, embedding in zip(log_texts, log_embeddings):
        score = cosine_similarity(question_embedding, embedding)
        scored_logs.append((text, score))

    scored_logs.sort(key=lambda x: x[1], reverse=True)
    top_logs = [log for log, _ in scored_logs[:top_k]]

    return "\n".join(f"{i + 1}. {log}" for i, log in enumerate(top_logs))


def parse_log(line):
    parts = line.split(":")

    if len(parts) >= 2:
        level = parts[0].strip()
        message = parts[1].strip()
    else:
        level = "UNKNOWN"
        message = line.strip()

    return {
        "level": level,
        "message": message
    }


def get_log_stats():
    if not parsed_logs:
        return {}

    levels = [log["level"] for log in parsed_logs]
    return dict(Counter(levels))


def compute_confidence():
    if not parsed_logs:
        return 0.0

    clusters = get_semantic_root_causes()

    if not clusters:
        return 0.0

    total_logs = len(parsed_logs)
    top_cluster_count = clusters[0]["count"]
    num_clusters = len(clusters)

    base_conf = top_cluster_count / total_logs
    penalty = min(num_clusters / 10, 0.5)
    confidence = base_conf * (1 - penalty)
    return round(confidence, 2)


def build_messages(question, history):
    context = get_relevant_logs_embeddings(question)
    stats = get_log_stats()
    root_causes = get_root_causes()
    semantic_causes = get_semantic_root_causes()
    confidence = compute_confidence()
    transitions = get_top_transitions()
    max_history = 5

    history_messages = [
        {
            "role": entry.get("role", "user"),
            "content": entry.get("content", "")
        }
        for entry in (history or [])[-max_history:]
        if entry.get("content")
    ]

    system_msg = {
        "role": "system",
        "content": (
            "Du bist ein Experte für Fahrzeug-Logs.\n"
            "Analysiere Logs präzise und faktenbasiert.\n"
            "Nutze nur die gegebenen Informationen.\n\n"
            "Antworte strukturiert:\n"
            "Problem:\n"
            "Ursache:\n"
            "Lösung:\n"
            "Confidence:\n"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Log Statistiken:\n{stats}\n\n"
            f"Häufigste Probleme:\n{root_causes}\n\n"
            f"Semantische Ursachen:\n{semantic_causes}\n\n"
            f"Confidence Score: {confidence}\n\n"
            f"Relevante Logs:\n{context}\n\n"
            f"Frage:\n{question}"
            f"Typische Fehler-Abfolgen:\n{transitions}\n\n"
        )
    }

    messages = [system_msg]
    messages.extend(history_messages)
    messages.append(user_msg)

    return messages


def get_root_causes(top_k=3):
    if not parsed_logs:
        return []

    messages = [log["message"] for log in parsed_logs]

    counter = Counter(messages)

    most_common = counter.most_common(top_k)

    return most_common


def get_semantic_root_causes(threshold=0.8):
    if not parsed_logs:
        return []

    clusters = []

    for log in parsed_logs:
        message = log["message"]
        emb = get_embedding(message)

        found_cluster = False

        for cluster in clusters:
            similarity = cosine_similarity(emb, cluster["embedding"])

            if similarity > threshold:
                cluster["messages"].append(message)
                cluster["count"] += 1
                found_cluster = True
                break

        if not found_cluster:
            clusters.append({
                "embedding": emb,
                "messages": [message],
                "count": 1
            })

    clusters.sort(key=lambda x: x["count"], reverse=True)
    result = []

    for cluster in clusters[:3]:
        result.append({
            "example": cluster["messages"][0],
            "count": cluster["count"]
        })

    return result


def get_error_transitions():
    transitions = {}

    for i in range(len(parsed_logs) - 1):
        current = parsed_logs[i]["message"]
        next_log = parsed_logs[i + 1]["message"]

        key = (current, next_log)

        if key not in transitions:
            transitions[key] = 0

        transitions[key] += 1

    return transitions


def get_top_transitions(top_k=3):
    transitions = get_error_transitions()
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)

    result = []
    for (current, next_log), count in sorted_transitions[:top_k]:
        result.append({
            "from": current,
            "to": next_log,
            "count": count
        })
    return result


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
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/ask_ai")
def ask_ai(request: ChatRequest):
    messages = build_messages(request.question, request.history)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages
    )

    answer = response.choices[0].message.content
    return {"answer": answer}


@app.post("/upload_logs")
def upload_logs(file: UploadFile = File(...)):
    global log_embeddings, log_texts, parsed_logs

    log_embeddings = []
    log_texts = []
    parsed_logs = []

    content = file.file.read().decode("utf-8")
    lines = content.splitlines()

    for line in lines:
        log_text = line.strip()

        if log_text:
            parsed_logs.append(parse_log(log_text))

            embedding = get_embedding(log_text)
            log_texts.append(log_text)
            log_embeddings.append(embedding)

    return {"message": f"{len(log_texts)} logs uploaded successfully."}


def initialize_embeddings(filepath):
    global log_embeddings, log_texts, parsed_logs

    log_embeddings = []
    log_texts = []
    parsed_logs = []

    with open(filepath, "r") as file:
        for line in file:
            log_text = line.strip()

            if log_text:
                parsed_logs.append(parse_log(log_text))

                embedding = get_embedding(log_text)
                log_texts.append(log_text)
                log_embeddings.append(embedding)


initialize_embeddings("log.txt")
