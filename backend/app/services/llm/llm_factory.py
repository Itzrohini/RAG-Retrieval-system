from loguru import logger
from app.config import settings


def get_llm_client():
    """
    Returns the correct LLM client based on LLM_PROVIDER in .env.
    Switch between gemini and ollama by changing one line in .env.
    """
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        logger.info("Using LLM provider: Gemini (cloud)")
        from app.services.llm.gemini_client import GeminiClient
        return GeminiClient()

    elif provider == "ollama":
        logger.info("Using LLM provider: Ollama (local)")
        from app.services.llm.ollama_client import OllamaClient
        return OllamaClient()

    else:
        logger.warning(f"Unknown LLM provider '{provider}', defaulting to Gemini")
        from app.services.llm.gemini_client import GeminiClient
        return GeminiClient()