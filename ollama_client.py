"""
Ollama client for LLM-powered intelligence features.

Handles:
- Text embeddings via nomic-embed-text (or user-configured model)
- Smart summarization
- Context-aware suggestions
- Natural language Q&A over project knowledge
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Default models — user can override via config
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    """
    Async Ollama HTTP client for embeddings and chat completions.

    Falls back gracefully when Ollama is unavailable — Nexus still works
    for all CRUD operations, just without AI-powered features.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._available: Optional[bool] = None

    async def close(self) -> None:
        await self._client.aclose()

    async def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        if self._available is not None:
            return self._available
        try:
            resp = await self._client.get("/api/tags")
            self._available = resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            self._available = False
            logger.warning("Ollama not available at %s — AI features disabled", self.base_url)
        return self._available

    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            resp = await self._client.get("/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float] | None:
        """Generate embedding vector for text using the embed model."""
        if not await self.is_available():
            return None

        try:
            resp = await self._client.post(
                "/api/embed",
                json={"model": self.embed_model, "input": text},
            )
            if resp.status_code == 200:
                data = resp.json()
                # Handle both single and batch responses
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings[0]
                # Fallback for older Ollama versions
                return data.get("embedding", None)
            else:
                logger.warning("Embed failed (status %d): %s", resp.status_code, resp.text[:200])
                return None
        except Exception as e:
            logger.warning("Embed error: %s", e)
            return None

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed multiple texts. Returns list of embeddings (None for failures)."""
        results = []
        for text in texts:
            emb = await self.embed(text)
            results.append(emb)
        return results

    # ------------------------------------------------------------------
    # Chat / Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str | None:
        """Generate text completion using the chat model."""
        if not await self.is_available():
            return None

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = await self._client.post(
                "/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("message", {}).get("content", "")
            else:
                logger.warning("Generate failed (status %d): %s", resp.status_code, resp.text[:200])
                return None
        except Exception as e:
            logger.warning("Generate error: %s", e)
            return None

    # ------------------------------------------------------------------
    # High-level intelligence methods
    # ------------------------------------------------------------------

    async def summarize(self, content: str, max_length: int = 300) -> str:
        """Generate a concise summary of content."""
        result = await self.generate(
            prompt=f"Summarize the following content in {max_length} characters or fewer. "
                   f"Be concise and capture the key points:\n\n{content[:4000]}",
            system="You are a precise technical summarizer. Output only the summary, no preamble.",
            temperature=0.2,
            max_tokens=512,
        )
        return result or content[:max_length]

    async def suggest_next_steps(
        self, project_context: str, current_state: str
    ) -> list[str]:
        """Suggest actionable next steps based on project context."""
        result = await self.generate(
            prompt=f"Given this project context:\n{project_context[:3000]}\n\n"
                   f"And current state:\n{current_state[:2000]}\n\n"
                   f"Suggest 3-5 specific, actionable next steps. "
                   f"Return as a JSON array of strings.",
            system="You are a senior engineering advisor. Return only a JSON array of actionable next steps.",
            temperature=0.4,
            max_tokens=1024,
        )
        if result:
            try:
                # Try to extract JSON array from response
                result = result.strip()
                if result.startswith("["):
                    return json.loads(result)
                # Try to find array in response
                start = result.find("[")
                end = result.rfind("]")
                if start >= 0 and end > start:
                    return json.loads(result[start:end + 1])
            except json.JSONDecodeError:
                pass
            # Fallback: split by newlines
            return [line.strip().lstrip("0123456789.-) ") for line in result.split("\n") if line.strip()]
        return ["Review existing specs and update task statuses",
                "Check for any incomplete features",
                "Consider adding tests for recent changes"]

    async def answer_question(self, question: str, context: str) -> str:
        """Answer a question using project context."""
        result = await self.generate(
            prompt=f"Based on the following project context, answer the question.\n\n"
                   f"Context:\n{context[:6000]}\n\n"
                   f"Question: {question}",
            system="You are a helpful project assistant. Answer based only on the provided context. "
                   "If the answer isn't in the context, say so clearly.",
            temperature=0.3,
            max_tokens=1024,
        )
        return result or "Unable to generate answer — Ollama may not be available."

    async def analyze_codebase_description(
        self, file_list: str, sample_content: str
    ) -> dict[str, str]:
        """Analyze a codebase and generate project description fields."""
        result = await self.generate(
            prompt=f"Analyze this codebase and generate project documentation.\n\n"
                   f"Files:\n{file_list[:3000]}\n\n"
                   f"Sample content:\n{sample_content[:3000]}\n\n"
                   f"Return a JSON object with keys: vision, tech_stack, summary",
            system="You are a senior software architect analyzing a codebase. "
                   "Return valid JSON with vision, tech_stack, and summary fields. "
                   "Each should be detailed markdown text.",
            temperature=0.3,
            max_tokens=2048,
        )
        if result:
            try:
                result = result.strip()
                start = result.find("{")
                end = result.rfind("}")
                if start >= 0 and end > start:
                    return json.loads(result[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "vision": "Project vision pending analysis",
            "tech_stack": "Tech stack pending analysis",
            "summary": "Summary pending analysis",
        }

    async def health_check(self) -> dict[str, Any]:
        """Return detailed health information."""
        available = await self.is_available()
        info: dict[str, Any] = {
            "available": available,
            "base_url": self.base_url,
            "chat_model": self.chat_model,
            "embed_model": self.embed_model,
        }
        if available:
            info["models"] = await self.list_models()
        return info
