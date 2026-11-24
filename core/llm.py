from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from core.config import get_settings, Settings


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """
    Build and configure a dual-compatible ChatOpenAI model instance ($0 stack).
    Seamlessly operates against local Ollama or Google AI Studio via environment settings.
    """
    cfg = settings or get_settings()
    return ChatOpenAI(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        temperature=0.0,
        timeout=120.0,
        max_retries=1,
    )
