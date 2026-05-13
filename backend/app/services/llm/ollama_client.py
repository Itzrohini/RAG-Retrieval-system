import httpx
import json
from loguru import logger
from typing import AsyncGenerator, List, Dict, Any
from app.config import settings


class OllamaClient:
    """
    Streams answers from a locally running Ollama LLM.
    Ollama runs Mistral 7B or LLaMA 3 on your machine — no API key needed.
    """

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def _build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """Assemble the RAG prompt with retrieved context."""
        context_text = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('file_name', 'unknown')}, "
            f"Page: {c['metadata'].get('page_number', 'N/A')}]\n{c['text']}"
            for c in context_chunks
        )

        system = (
            "You are a precise document assistant. "
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say 'I could not find this in the documents.' "
            "Always cite the source file and page number when possible."
        )

        prompt = f"{system}\n\n"
        if memory_context:
            prompt += f"{memory_context}\n\n"
        prompt += f"Context:\n{context_text}\n\n"
        prompt += f"Question: {question}\n\nAnswer:"
        return prompt

    async def stream_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """Yields answer tokens one by one as they stream from Ollama."""
        prompt = self._build_prompt(question, context_chunks, memory_context)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,   # low temp = more factual
                "top_p": 0.9,
                "num_predict": 1024,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.error(
                            "Ollama HTTP %s error: %s",
                            response.status_code,
                            body.decode("utf-8", errors="replace"),
                        )
                        response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            yield (
                "Error: Could not connect to Ollama. "
                "Make sure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            yield f"Error generating answer: {str(e)}"

    async def get_full_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """Non-streaming version — returns complete answer."""
        tokens = []
        async for token in self.stream_answer(question, context_chunks, memory_context):
            tokens.append(token)
        return "".join(tokens)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False