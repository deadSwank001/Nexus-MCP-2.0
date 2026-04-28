"""
Project management — create, load, list, archive, analyze projects.

This replaces foundry-mcp's create_project/load_project/list_projects/analyze_project
with AI-enhanced versions that generate summaries and track richer metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.storage.database import Database
from nexus_mcp.storage.vector_store import VectorStore


class ProjectManager:
    """Manages project lifecycle with optional AI enhancement."""

    def __init__(self, db: Database, ollama: OllamaClient, vectors: VectorStore):
        self._db = db
        self._ollama = ollama
        self._vectors = vectors

    async def create_project(
        self,
        name: str,
        vision: str = "",
        tech_stack: str = "",
        summary: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new project with metadata."""
        # Validate name
        name = name.strip().lower().replace(" ", "-")
        if not name:
            raise ValueError("Project name cannot be empty")

        # Check for duplicates
        existing = await self._db.get_project(name)
        if existing:
            raise ValueError(f"Project '{name}' already exists")

        project = await self._db.create_project(
            name=name,
            vision=vision,
            tech_stack=tech_stack,
            summary=summary,
            tags=tags,
        )

        # Generate embeddings for searchability
        await self._embed_project(project)

        spec_count = await self._db.get_project_spec_count(project["id"])

        return {
            "status": "created",
            "project": {
                "name": project["name"],
                "created_at": project["created_at"],
                "spec_count": spec_count,
                "has_vision": bool(project["vision"]),
                "has_tech_stack": bool(project["tech_stack"]),
            },
            "next_steps": [
                f"Create a spec: use create_spec with project_name='{name}'",
                f"Load full context: use load_project with project_name='{name}'",
                "Add documents: use create_document for design docs, ADRs, etc.",
            ],
        }

    async def load_project(
        self,
        name: str,
        include_specs: bool = True,
        summarize: bool = False,
        token_budget: int = 0,
    ) -> dict[str, Any]:
        """
        Load project context, optionally with AI summarization.

        If summarize=True and Ollama is available, returns condensed context
        optimized for LLM token windows.
        """
        project = await self._db.get_project(name)
        if not project:
            raise ValueError(f"Project '{name}' not found")

        result: dict[str, Any] = {
            "project": {
                "name": project["name"],
                "vision": project["vision"],
                "tech_stack": project["tech_stack"],
                "summary": project["summary"],
                "tags": json.loads(project["tags"]) if isinstance(project["tags"], str) else project["tags"],
                "created_at": project["created_at"],
                "updated_at": project["updated_at"],
            }
        }

        if include_specs:
            specs = await self._db.list_specs(name)
            result["specs"] = [
                {
                    "name": s["name"],
                    "feature_name": s["feature_name"],
                    "status": s["status"],
                    "created_at": s["created_at"],
                }
                for s in specs
            ]

        # AI summarization for token-constrained contexts
        if summarize and token_budget > 0:
            full_context = json.dumps(result, indent=2)
            if len(full_context) > token_budget * 4:  # ~4 chars per token
                summary = await self._ollama.summarize(
                    full_context, max_length=token_budget * 4
                )
                result["_condensed_summary"] = summary
                result["_token_budget_applied"] = True

        # Check for active session
        active_session = await self._db.get_active_session(name)
        if active_session:
            result["active_session"] = {
                "id": active_session["id"],
                "started_at": active_session["started_at"],
            }

        return result

    async def list_projects(
        self, include_archived: bool = False
    ) -> dict[str, Any]:
        """List all projects with metadata."""
        projects = await self._db.list_projects(include_archived=include_archived)
        project_list = []
        for p in projects:
            spec_count = await self._db.get_project_spec_count(p["id"])
            project_list.append({
                "name": p["name"],
                "summary": p["summary"][:200] if p["summary"] else "",
                "spec_count": spec_count,
                "tags": json.loads(p["tags"]) if isinstance(p["tags"], str) else p["tags"],
                "created_at": p["created_at"],
                "updated_at": p["updated_at"],
                "archived": bool(p["archived"]),
            })
        return {
            "projects": project_list,
            "total_count": len(project_list),
        }

    async def archive_project(self, name: str) -> dict[str, Any]:
        """Soft-delete a project (can be restored)."""
        success = await self._db.archive_project(name)
        if not success:
            raise ValueError(f"Project '{name}' not found or already archived")
        return {
            "status": "archived",
            "project_name": name,
            "message": f"Project '{name}' has been archived. Use list_projects with include_archived=true to see it.",
        }

    async def update_project(
        self, name: str, **fields
    ) -> dict[str, Any]:
        """Update project fields."""
        result = await self._db.update_project(name, **fields)
        if not result:
            raise ValueError(f"Project '{name}' not found")

        # Re-embed after update
        await self._embed_project(result)

        return {
            "status": "updated",
            "project_name": name,
            "updated_fields": list(fields.keys()),
        }

    async def analyze_codebase(
        self,
        name: str,
        file_list: str,
        sample_content: str = "",
    ) -> dict[str, Any]:
        """
        Create/update a project by analyzing codebase files.

        Uses Ollama to generate vision, tech_stack, and summary
        from file listing and sample content.
        """
        analysis = await self._ollama.analyze_codebase_description(
            file_list, sample_content
        )

        existing = await self._db.get_project(name)
        if existing:
            await self._db.update_project(
                name,
                vision=analysis.get("vision", ""),
                tech_stack=analysis.get("tech_stack", ""),
                summary=analysis.get("summary", ""),
            )
            status = "updated"
        else:
            await self._db.create_project(
                name=name,
                vision=analysis.get("vision", ""),
                tech_stack=analysis.get("tech_stack", ""),
                summary=analysis.get("summary", ""),
            )
            status = "created"

        return {
            "status": status,
            "project_name": name,
            "analysis": analysis,
            "ai_generated": await self._ollama.is_available(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_project(self, project: dict[str, Any]) -> None:
        """Generate and store embeddings for project content."""
        if not await self._ollama.is_available():
            return

        content = f"{project.get('name', '')} {project.get('vision', '')} {project.get('tech_stack', '')} {project.get('summary', '')}"
        embedding = await self._ollama.embed(content[:2000])
        if embedding:
            await self._vectors.clear("project", project["id"])
            await self._vectors.store("project", project["id"], content[:500], embedding)
