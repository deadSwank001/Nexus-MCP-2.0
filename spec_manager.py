"""
Specification management — create, load, update, delete, diff specs.

Replaces foundry-mcp's 1336-line edit engine with a clean, simple approach:
- Direct content replacement (no complex selector/command system)
- Version tracking via updated_at timestamps
- AI-powered task suggestions
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.storage.database import Database
from nexus_mcp.storage.vector_store import VectorStore


def _generate_spec_name(feature_name: str) -> str:
    """Generate a timestamped spec name from a feature name."""
    now = datetime.now(timezone.utc)
    clean_name = feature_name.strip().lower().replace(" ", "_").replace("-", "_")
    # Remove non-alphanumeric/underscore chars
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    return f"{now.strftime('%Y%m%d_%H%M%S')}_{clean_name}"


class SpecManager:
    """Manages feature specifications within projects."""

    def __init__(self, db: Database, ollama: OllamaClient, vectors: VectorStore):
        self._db = db
        self._ollama = ollama
        self._vectors = vectors

    async def create_spec(
        self,
        project_name: str,
        feature_name: str,
        spec_content: str = "",
        tasks_content: str = "",
        notes_content: str = "",
        auto_suggest_tasks: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new specification.

        If auto_suggest_tasks is True and Ollama is available,
        generates suggested task breakdown from the spec content.
        """
        # Validate project exists
        project = await self._db.get_project(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found")

        spec_name = _generate_spec_name(feature_name)

        # AI-powered task suggestion
        if auto_suggest_tasks and spec_content and not tasks_content:
            suggested = await self._suggest_tasks(spec_content, feature_name)
            if suggested:
                tasks_content = suggested

        spec = await self._db.create_spec(
            project_name=project_name,
            name=spec_name,
            feature_name=feature_name,
            spec_content=spec_content,
            tasks_content=tasks_content,
            notes_content=notes_content,
        )
        if not spec:
            raise ValueError(f"Failed to create spec in project '{project_name}'")

        # Generate embeddings
        await self._embed_spec(spec)

        return {
            "status": "created",
            "project_name": project_name,
            "spec_name": spec_name,
            "feature_name": feature_name,
            "created_at": spec["created_at"],
            "has_tasks": bool(tasks_content),
            "has_notes": bool(notes_content),
            "next_steps": [
                f"Load spec: use load_spec with project_name='{project_name}' and spec_name='{spec_name}'",
                f"Update spec: use update_spec to modify content",
                f"View all specs: use list_specs with project_name='{project_name}'",
            ],
        }

    async def load_spec(
        self,
        project_name: str,
        spec_name: str,
    ) -> dict[str, Any]:
        """Load a specification with its content."""
        spec = await self._db.get_spec(project_name, spec_name)
        if not spec:
            # Try fuzzy matching
            all_specs = await self._db.list_specs(project_name)
            matches = self._fuzzy_match(spec_name, all_specs)
            if len(matches) == 1:
                spec = await self._db.get_spec(project_name, matches[0]["name"])
            elif len(matches) > 1:
                return {
                    "status": "ambiguous",
                    "message": f"Multiple specs match '{spec_name}'",
                    "candidates": [
                        {"name": m["name"], "feature_name": m["feature_name"]}
                        for m in matches
                    ],
                }
            else:
                raise ValueError(
                    f"Spec '{spec_name}' not found in project '{project_name}'"
                )

        # Load project summary for context
        project = await self._db.get_project(project_name)
        project_summary = project["summary"] if project else ""

        return {
            "project_name": project_name,
            "project_summary": project_summary,
            "spec": {
                "name": spec["name"],
                "feature_name": spec["feature_name"],
                "status": spec["status"],
                "created_at": spec["created_at"],
                "updated_at": spec["updated_at"],
                "content": {
                    "spec": spec["spec_content"],
                    "tasks": spec["tasks_content"],
                    "notes": spec["notes_content"],
                },
            },
        }

    async def list_specs(self, project_name: str) -> dict[str, Any]:
        """List all specs in a project."""
        specs = await self._db.list_specs(project_name)
        return {
            "project_name": project_name,
            "specs": [
                {
                    "name": s["name"],
                    "feature_name": s["feature_name"],
                    "status": s["status"],
                    "created_at": s["created_at"],
                    "updated_at": s["updated_at"],
                }
                for s in specs
            ],
            "total_count": len(specs),
        }

    async def update_spec(
        self,
        project_name: str,
        spec_name: str,
        spec_content: str | None = None,
        tasks_content: str | None = None,
        notes_content: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        Update spec content directly.

        Unlike foundry-mcp's complex edit command system (1336 lines of match arms),
        Nexus uses simple direct content replacement. The calling AI agent is already
        great at text manipulation — we don't need to replicate a text editor.
        """
        spec = await self._db.get_spec(project_name, spec_name)
        if not spec:
            raise ValueError(f"Spec '{spec_name}' not found in project '{project_name}'")

        fields: dict[str, Any] = {}
        if spec_content is not None:
            fields["spec_content"] = spec_content
        if tasks_content is not None:
            fields["tasks_content"] = tasks_content
        if notes_content is not None:
            fields["notes_content"] = notes_content
        if status is not None:
            if status not in ("active", "completed", "on_hold", "cancelled"):
                raise ValueError(f"Invalid status: {status}")
            fields["status"] = status

        if not fields:
            raise ValueError("No fields to update")

        updated = await self._db.update_spec(project_name, spec_name, **fields)
        if not updated:
            raise ValueError("Update failed")

        # Re-embed after update
        await self._embed_spec(updated)

        return {
            "status": "updated",
            "project_name": project_name,
            "spec_name": spec_name,
            "updated_fields": list(fields.keys()),
        }

    async def delete_spec(
        self,
        project_name: str,
        spec_name: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Delete a spec with confirmation."""
        if not confirm:
            return {
                "status": "confirmation_required",
                "message": f"To delete spec '{spec_name}', call again with confirm=true",
                "spec_name": spec_name,
                "project_name": project_name,
            }

        success = await self._db.delete_spec(project_name, spec_name)
        if not success:
            raise ValueError(f"Spec '{spec_name}' not found in project '{project_name}'")

        # Clean up embeddings
        await self._vectors.clear("spec", 0)  # Will be cleaned up properly

        return {
            "status": "deleted",
            "project_name": project_name,
            "spec_name": spec_name,
        }

    async def diff_spec(
        self,
        project_name: str,
        spec_name: str,
        new_spec_content: str | None = None,
        new_tasks_content: str | None = None,
        new_notes_content: str | None = None,
    ) -> dict[str, Any]:
        """Show differences between current and proposed content."""
        spec = await self._db.get_spec(project_name, spec_name)
        if not spec:
            raise ValueError(f"Spec '{spec_name}' not found")

        diffs = {}
        if new_spec_content is not None:
            diffs["spec"] = {
                "current_length": len(spec["spec_content"]),
                "new_length": len(new_spec_content),
                "changed": spec["spec_content"] != new_spec_content,
            }
        if new_tasks_content is not None:
            diffs["tasks"] = {
                "current_length": len(spec["tasks_content"]),
                "new_length": len(new_tasks_content),
                "changed": spec["tasks_content"] != new_tasks_content,
            }
        if new_notes_content is not None:
            diffs["notes"] = {
                "current_length": len(spec["notes_content"]),
                "new_length": len(new_notes_content),
                "changed": spec["notes_content"] != new_notes_content,
            }

        return {
            "project_name": project_name,
            "spec_name": spec_name,
            "diffs": diffs,
            "any_changes": any(d["changed"] for d in diffs.values()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_match(
        self, query: str, specs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Simple fuzzy matching by substring on name and feature_name."""
        query_lower = query.lower()
        matches = []
        for s in specs:
            if (
                query_lower in s["name"].lower()
                or query_lower in s["feature_name"].lower()
            ):
                matches.append(s)
        return matches

    async def _suggest_tasks(self, spec_content: str, feature_name: str) -> str | None:
        """Use Ollama to suggest a task breakdown."""
        result = await self._ollama.generate(
            prompt=f"Given this feature specification for '{feature_name}':\n\n"
                   f"{spec_content[:3000]}\n\n"
                   f"Generate a markdown task list (using - [ ] checkboxes) "
                   f"breaking this feature into implementable tasks. "
                   f"Order them logically. Include 5-10 tasks.",
            system="You are a senior engineer creating implementation task lists. "
                   "Output only the markdown task list, nothing else.",
            temperature=0.3,
        )
        return result

    async def _embed_spec(self, spec: dict[str, Any]) -> None:
        """Generate and store embeddings for spec content."""
        if not await self._ollama.is_available():
            return

        content = (
            f"{spec.get('feature_name', '')} "
            f"{spec.get('spec_content', '')} "
            f"{spec.get('tasks_content', '')}"
        )
        embedding = await self._ollama.embed(content[:2000])
        if embedding:
            await self._vectors.clear("spec", spec["id"])
            await self._vectors.store("spec", spec["id"], content[:500], embedding)
