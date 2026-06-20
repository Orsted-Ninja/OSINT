from langchain_ollama import ChatOllama
from backend.core.config import settings

def get_llm(model: str = None):
    actual_model = model or settings.OLLAMA_MODEL
    return ChatOllama(
        model=actual_model,
        temperature=0,
        streaming=False,
        top_p=0.9,
        num_ctx=8192 #32768 for larger context window
    )