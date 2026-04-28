"""
Semantic search engine — search across all projects using embeddings + FTS5.

Combines vector similarity (via Ollama embeddings) with SQLite FTS5 full-text
search for hybrid retrieval. Not present in foundry-mcp at all.
"""

from __future__ import annotations

from typing import Any

from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.storage.database import Database
from nexus_mcp.storage.vector_store import VectorStore


class SearchEngine:
    """Hybrid semantic + full-text search across all project content."""

    def __init__(self, db: Database, ollama: OllamaClient, vectors: VectorStore):
        self._db = db
        self._ollama = ollama
        self._vectors = vectors

    async def search(
        self,
        query: str,
        top_k: int = 10,
        project_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Search across all projects using hybrid retrieval.

        1. FTS5 keyword search (always available)
        2. Vector similarity search (when Ollama is available)
        3. Merge and deduplicate results
        """
        results = []

        # FTS5 search — always works
        fts_results = await self._db.search_fts(query, limit=top_k)
        for r in fts_results:
            results.append({
                "source_type": r["source_type"],
                "source_id": r["source_id"],
                "title": r["title"],
                "snippet": r.get("snippet", ""),
                "match_type": "keyword",
                "score": abs(float(r.get("rank", 0))),
            })

        # Vector search — only when Ollama is available
        query_embedding = await self._ollama.embed(query)
        if query_embedding:
            vec_results = await self._vectors.search(
                query_embedding, top_k=top_k, min_similarity=0.3
            )
            for r in vec_results:
                # Check for duplicates
                is_dup = any(
                    existing["source_type"] == r["source_type"]
                    and str(existing["source_id"]) == str(r["source_id"])
                    for existing in results
                )
                if not is_dup:
                    results.append({
                        "source_type": r["source_type"],
                        "source_id": r["source_id"],
                        "title": r["chunk_text"][:100],
                        "snippet": r["chunk_text"][:200],
                        "match_type": "semantic",
                        "score": r["similarity"],
                    })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "results": results[:top_k],
            "total_matches": len(results),
            "search_methods": ["keyword"] + (["semantic"] if query_embedding else []),
        }

    async def find_related(
        self,
        project_name: str,
        spec_name: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Find content related to a project or spec across all projects."""
        # Build context from the source
        if spec_name:
            spec = await self._db.get_spec(project_name, spec_name)
            if not spec:
                raise ValueError(f"Spec '{spec_name}' not found")
            context = f"{spec['feature_name']} {spec['spec_content'][:1000]}"
        else:
            project = await self._db.get_project(project_name)
            if not project:
                raise ValueError(f"Project '{project_name}' not found")
            context = f"{project['name']} {project['vision'][:500]} {project['summary'][:500]}"

        # Search for related content
        return await self.search(context[:500], top_k=top_k)

    async def ask(self, question: str, project_name: str | None = None) -> dict[str, Any]:
        """
        Answer a natural language question using project knowledge.

        Retrieves relevant context via search, then uses Ollama to generate answer.
        """
        # Gather context
        search_results = await self.search(question, top_k=5)
        context_parts = []

        if project_name:
            project = await self._db.get_project(project_name)
            if project:
                context_parts.append(f"Project: {project['name']}\nVision: {project['vision']}\nTech: {project['tech_stack']}\nSummary: {project['summary']}")

        for r in search_results["results"]:
            context_parts.append(f"[{r['source_type']}] {r['title']}: {r['snippet']}")

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = await self._ollama.answer_question(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources_used": len(search_results["results"]),
            "project_context": project_name,
        }
