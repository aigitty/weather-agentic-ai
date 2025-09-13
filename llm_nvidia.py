# llm_nvidia.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY  = os.getenv("NVIDIA_API_KEY")
MODEL    = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

if not API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is missing. Add it to your .env")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def chat_text(prompt: str, system: str = "You are a helpful weather assistant.", temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Return plain text from the LLM."""
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()
