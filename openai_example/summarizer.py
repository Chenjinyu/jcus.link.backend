# summarizer.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize(upload_text, related_docs):
    context = "\n\n".join([doc[0] for doc in related_docs])

    prompt = f"""
You are an AI assistant.

User uploaded this file content:
{upload_text}

Similar related documents from vector DB:
{context}

Write a concise summary combining both.
    """

    result = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )

    return result.choices[0].message.content
