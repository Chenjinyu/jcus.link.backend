# embeddings.py
import os

import google.generativeai as genai
import ollama
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_embedding(text: str, model_name: str):
    if model_name == "openai":
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return resp.data[0].embedding

    if model_name == "google":
        model = genai.GenerativeModel("text-embedding-004")
        return model.embed_content(content=text)["embedding"]

    if model_name == "ollama":
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

    raise ValueError("Unsupported embedding model")
