"""
Session management — track development sessions for continuity.

This is entirely NEW functionality not present in foundry-mcp.
"""

from __future__ import annotations

import json
from typing import Any

from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.storage.database import Database


class SessionManager:
    """Tracks development sessions for cross-session continuity."""

    def __init__(self, db: Database, ollama: OllamaClient):
        self._db = db
        self._ollama = ollama

    async def start_session(self, project_name: str) -> dict[str, Any]:
        active = await self._db.get_active_session(project_name)
        if active:
            return {
                "status": "existing_session",
                "session": {"id": active["id"], "started_at": active["started_at"]},
            }
        session = await self._db.start_session(project_name)
        if not session:
            raise ValueError(f"Project '{project_name}' not found")
        recent = await self._db.get_recent_sessions(project_name, limit=3)
        recent_summaries = [
            {"session_id": r["id"], "summary": r["summary"][:200]}
            for r in recent if r["id"] != session["id"] and r["summary"]
        ]
        return {"status": "started", "session": session, "recent_sessions": recent_summaries}

    async def end_session(self, session_id: int, summary: str = "", decisions: list[str] | None = None) -> dict[str, Any]:
        if not summary:
            summary = "Session completed"
        success = await self._db.end_session(session_id, summary, decisions)
        if not success:
            raise ValueError(f"Session {session_id} not found or already ended")
        return {"status": "ended", "session_id": session_id, "summary": summary}

    async def resume_session(self, project_name: str) -> dict[str, Any]:
        recent = await self._db.get_recent_sessions(project_name, limit=1)
        new_session = await self.start_session(project_name)
        if recent and recent[0].get("summary"):
            last = recent[0]
            new_session["previous_session"] = {
                "id": last["id"], "summary": last["summary"],
                "decisions": json.loads(last["decisions"]) if isinstance(last["decisions"], str) else last.get("decisions", []),
            }
        return new_session
