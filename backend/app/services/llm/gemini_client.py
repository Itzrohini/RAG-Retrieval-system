import google.generativeai as genai
from loguru import logger
from typing import AsyncGenerator, List, Dict, Any
import asyncio
from app.config import settings


class GeminiClient:
    """
    Gemini API client.
    Handles two modes:
    1. RAG mode  — answer from PDF context
    2. General mode — answer from general knowledge
    """

    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in .env file. "
                "Get your key from https://aistudio.google.com"
            )
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model
        print("MODEL =", self.model_name)
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 1024,
                "top_p": 0.9,
            },
        )
        logger.info(f"Gemini client ready: {self.model_name}")

    def _is_general_question(self, question: str, context_chunks: List[Dict]) -> bool:
        """
        Detect if question is general knowledge vs document-specific.
        Only treat as general if NO chunks were retrieved at all.
        The retriever already filters by relevance — trust its output.
        """
        # No chunks = nothing found in docs = general question
        if not context_chunks:
            return True

        # Check hardcoded greetings / meta questions that are never in docs
        general_keywords = [
            "hello", "hi", "how are you",
            "what can you do", "who are you", "what is your name",
            "tell me about yourself",
        ]
        q_lower = question.lower().strip()
        if any(kw == q_lower for kw in general_keywords):
            return True

        # We have chunks — trust the retriever, use RAG mode
        # DO NOT check score here — RRF scores are tiny (max ~0.016)
        # and a threshold of 0.02 would always trigger general mode
        return False

    def _build_rag_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        """Prompt for document-based questions."""
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5]):
            meta = chunk.get("metadata", {})
            file_name = meta.get("file_name", "unknown")
            page = meta.get("page_number", "?")
            text = chunk.get("text", "")[:600]
            context_parts.append(
                f"[Source {i+1}: {file_name}, Page {page}]\n{text}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful study assistant. Use the document context below to answer the question.

Rules:
- Answer ONLY from the context provided below
- Use **bold** for key terms and definitions
- End with: "Source: [filename], Page [number]"
- If the answer spans multiple sources, cite all of them
- If the context does not contain the answer, say exactly: "This topic is not covered in the uploaded documents."
- Do NOT use your general knowledge — only use what is in the context

Context:
{context_text}

{f"Prior conversation:{memory_context}" if memory_context else ""}

Question: {question}

Answer:"""
        return prompt

    def _build_general_prompt(self, question: str, memory_context: str = "") -> str:
        """Prompt for general knowledge questions."""
        return f"""You are a helpful assistant. Answer this question clearly and helpfully.
Use **bold** for important terms.
Be concise but complete.
Since no relevant documents were found, answer from general knowledge and mention this is general knowledge, not from uploaded documents.

{f"Prior conversation:{memory_context}" if memory_context else ""}

Question: {question}

Answer:"""

    async def stream_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> AsyncGenerator[str, None]:
        """Stream answer — auto-detects RAG vs general mode."""

        is_general = self._is_general_question(question, context_chunks)

        if is_general:
            prompt = self._build_general_prompt(question, memory_context)
            logger.info("Mode: GENERAL knowledge (no chunks retrieved)")
        else:
            prompt = self._build_rag_prompt(question, context_chunks, memory_context)
            logger.info(f"Mode: RAG from documents ({len(context_chunks)} chunks)")

        try:
            tokens = await asyncio.to_thread(self._stream_sync, prompt)
            for token in tokens:
                yield token

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini error: {error_msg}")

            if "404" in error_msg or "not found" in error_msg.lower():
                yield (
                    f"Error: Gemini model '{self.model_name}' not found. "
                    f"Update GEMINI_MODEL in .env to: gemini-2.0-flash"
                )
            elif "API_KEY" in error_msg or "api_key" in error_msg:
                yield "Error: Invalid Gemini API key. Check GEMINI_API_KEY in your .env file."
            elif "quota" in error_msg.lower():
                yield "Gemini API quota exceeded. Please wait a minute and try again."
            else:
                yield f"Error from Gemini: {error_msg}"

    def _stream_sync(self, prompt: str) -> List[str]:
        """Synchronous call — runs in thread pool."""
        tokens = []
        response = self.model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                tokens.append(chunk.text)
        return tokens

    async def get_full_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        memory_context: str = "",
    ) -> str:
        tokens = []
        async for token in self.stream_answer(question, context_chunks, memory_context):
            tokens.append(token)
        return "".join(tokens)

    async def health_check(self) -> bool:
        try:
            resp = await asyncio.to_thread(
                self.model.generate_content, "Say OK in one word"
            )
            return bool(resp.text)
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False