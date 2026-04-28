"""
SQLite database layer with FTS5 full-text search.

This replaces foundry-mcp's flat-file approach with a proper queryable database
while still maintaining human-readable file storage for content.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nexus_dir() -> Path:
    """Return ~/.nexus, creating it if needed."""
    p = Path.home() / ".nexus"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _db_path() -> Path:
    return _nexus_dir() / "nexus.db"


def _content_dir() -> Path:
    """Content files live alongside the DB for backup-friendliness."""
    p = _nexus_dir() / "content"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
-- Core project table
CREATE TABLE IF NOT EXISTS projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    vision      TEXT    NOT NULL DEFAULT '',
    tech_stack  TEXT    NOT NULL DEFAULT '',
    summary     TEXT    NOT NULL DEFAULT '',
    tags        TEXT    NOT NULL DEFAULT '[]',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL,
    archived    INTEGER NOT NULL DEFAULT 0
);

-- Specifications within a project
CREATE TABLE IF NOT EXISTS specs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,
    feature_name  TEXT    NOT NULL,
    spec_content  TEXT    NOT NULL DEFAULT '',
    tasks_content TEXT    NOT NULL DEFAULT '',
    notes_content TEXT    NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'active',
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL,
    UNIQUE(project_id, name)
);

-- Arbitrary documents (ADRs, design docs, runbooks, etc.)
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    doc_type    TEXT    NOT NULL DEFAULT 'general',
    title       TEXT    NOT NULL,
    content     TEXT    NOT NULL DEFAULT '',
    tags        TEXT    NOT NULL DEFAULT '[]',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

-- Session tracking for development continuity
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    summary     TEXT    NOT NULL DEFAULT '',
    decisions   TEXT    NOT NULL DEFAULT '[]',
    topics      TEXT    NOT NULL DEFAULT '[]',
    status      TEXT    NOT NULL DEFAULT 'active'
);

-- Embedding vectors for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT    NOT NULL,
    source_id   INTEGER NOT NULL,
    chunk_text  TEXT    NOT NULL,
    embedding   BLOB   NOT NULL,
    created_at  TEXT    NOT NULL
);

-- Full-text search index
CREATE VIRTUAL TABLE IF NOT EXISTS fts_content USING fts5(
    source_type,
    source_id UNINDEXED,
    title,
    content,
    tags
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_specs_project ON specs(project_id);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """Async SQLite database with FTS5 full-text search."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or _db_path()
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    async def create_project(
        self,
        name: str,
        vision: str = "",
        tech_stack: str = "",
        summary: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        now = _utcnow()
        tags_json = json.dumps(tags or [])
        await self.db.execute(
            """INSERT INTO projects (name, vision, tech_stack, summary, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, vision, tech_stack, summary, tags_json, now, now),
        )
        await self.db.commit()
        return await self.get_project(name)

    async def get_project(self, name: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM projects WHERE name = ? AND archived = 0", (name,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_projects(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM projects"
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY updated_at DESC"
        async with self.db.execute(query) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_project(self, name: str, **fields) -> dict[str, Any] | None:
        project = await self.get_project(name)
        if not project:
            return None

        fields["updated_at"] = _utcnow()
        if "tags" in fields and isinstance(fields["tags"], list):
            fields["tags"] = json.dumps(fields["tags"])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [name]
        await self.db.execute(
            f"UPDATE projects SET {set_clause} WHERE name = ?", values
        )
        await self.db.commit()
        return await self.get_project(name)

    async def archive_project(self, name: str) -> bool:
        result = await self.db.execute(
            "UPDATE projects SET archived = 1, updated_at = ? WHERE name = ? AND archived = 0",
            (_utcnow(), name),
        )
        await self.db.commit()
        return result.rowcount > 0

    async def get_project_spec_count(self, project_id: int) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) FROM specs WHERE project_id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    # ------------------------------------------------------------------
    # Spec CRUD
    # ------------------------------------------------------------------

    async def create_spec(
        self,
        project_name: str,
        name: str,
        feature_name: str,
        spec_content: str = "",
        tasks_content: str = "",
        notes_content: str = "",
    ) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        now = _utcnow()
        await self.db.execute(
            """INSERT INTO specs (project_id, name, feature_name, spec_content, tasks_content,
               notes_content, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project["id"], name, feature_name, spec_content, tasks_content, notes_content, now, now),
        )
        await self.db.commit()

        # Index in FTS
        async with self.db.execute(
            "SELECT id FROM specs WHERE project_id = ? AND name = ?",
            (project["id"], name)
        ) as cursor:
            spec_row = await cursor.fetchone()
            if spec_row:
                await self._index_fts("spec", spec_row[0], feature_name,
                                       f"{spec_content}\n{tasks_content}\n{notes_content}", "[]")

        return await self.get_spec(project_name, name)

    async def get_spec(self, project_name: str, spec_name: str) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        async with self.db.execute(
            "SELECT * FROM specs WHERE project_id = ? AND name = ?",
            (project["id"], spec_name),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_specs(self, project_name: str) -> list[dict[str, Any]]:
        project = await self.get_project(project_name)
        if not project:
            return []
        async with self.db.execute(
            "SELECT * FROM specs WHERE project_id = ? ORDER BY created_at DESC",
            (project["id"],),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def update_spec(
        self, project_name: str, spec_name: str, **fields
    ) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        fields["updated_at"] = _utcnow()
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [project["id"], spec_name]
        await self.db.execute(
            f"UPDATE specs SET {set_clause} WHERE project_id = ? AND name = ?",
            values,
        )
        await self.db.commit()
        return await self.get_spec(project_name, spec_name)

    async def delete_spec(self, project_name: str, spec_name: str) -> bool:
        project = await self.get_project(project_name)
        if not project:
            return False
        result = await self.db.execute(
            "DELETE FROM specs WHERE project_id = ? AND name = ?",
            (project["id"], spec_name),
        )
        await self.db.commit()
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Document CRUD
    # ------------------------------------------------------------------

    async def create_document(
        self,
        project_name: str,
        title: str,
        content: str = "",
        doc_type: str = "general",
        tags: list[str] | None = None,
    ) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        now = _utcnow()
        tags_json = json.dumps(tags or [])
        await self.db.execute(
            """INSERT INTO documents (project_id, doc_type, title, content, tags, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (project["id"], doc_type, title, content, tags_json, now, now),
        )
        await self.db.commit()

        # Index in FTS
        async with self.db.execute("SELECT last_insert_rowid()") as cursor:
            row = await cursor.fetchone()
            doc_id = row[0]
        await self._index_fts("document", doc_id, title, content, tags_json)

        return {"id": doc_id, "project_name": project_name, "title": title,
                "doc_type": doc_type, "created_at": now}

    async def list_documents(self, project_name: str) -> list[dict[str, Any]]:
        project = await self.get_project(project_name)
        if not project:
            return []
        async with self.db.execute(
            "SELECT * FROM documents WHERE project_id = ? ORDER BY updated_at DESC",
            (project["id"],),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Session tracking
    # ------------------------------------------------------------------

    async def start_session(self, project_name: str) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        now = _utcnow()
        await self.db.execute(
            """INSERT INTO sessions (project_id, started_at, status) VALUES (?, ?, 'active')""",
            (project["id"], now),
        )
        await self.db.commit()
        async with self.db.execute("SELECT last_insert_rowid()") as cursor:
            row = await cursor.fetchone()
            session_id = row[0]
        return {"id": session_id, "project_name": project_name, "started_at": now, "status": "active"}

    async def end_session(
        self, session_id: int, summary: str = "", decisions: list[str] | None = None
    ) -> bool:
        now = _utcnow()
        result = await self.db.execute(
            """UPDATE sessions SET ended_at = ?, summary = ?, decisions = ?, status = 'completed'
               WHERE id = ? AND status = 'active'""",
            (now, summary, json.dumps(decisions or []), session_id),
        )
        await self.db.commit()
        return result.rowcount > 0

    async def get_active_session(self, project_name: str) -> dict[str, Any] | None:
        project = await self.get_project(project_name)
        if not project:
            return None
        async with self.db.execute(
            "SELECT * FROM sessions WHERE project_id = ? AND status = 'active' ORDER BY started_at DESC LIMIT 1",
            (project["id"],),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def get_recent_sessions(self, project_name: str, limit: int = 5) -> list[dict[str, Any]]:
        project = await self.get_project(project_name)
        if not project:
            return []
        async with self.db.execute(
            "SELECT * FROM sessions WHERE project_id = ? ORDER BY started_at DESC LIMIT ?",
            (project["id"], limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Full-text search
    # ------------------------------------------------------------------

    async def _index_fts(
        self, source_type: str, source_id: int, title: str, content: str, tags: str
    ) -> None:
        # Remove old entry first
        await self.db.execute(
            "DELETE FROM fts_content WHERE source_type = ? AND source_id = ?",
            (source_type, str(source_id)),
        )
        await self.db.execute(
            "INSERT INTO fts_content (source_type, source_id, title, content, tags) VALUES (?, ?, ?, ?, ?)",
            (source_type, str(source_id), title, content, tags),
        )
        await self.db.commit()

    async def search_fts(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across all indexed content."""
        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')
        try:
            async with self.db.execute(
                """SELECT source_type, source_id, title,
                          snippet(fts_content, 3, '>>>', '<<<', '...', 64) as snippet,
                          rank
                   FROM fts_content
                   WHERE fts_content MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]
        except Exception:
            # Fallback: try simpler query
            async with self.db.execute(
                """SELECT source_type, source_id, title,
                          snippet(fts_content, 3, '>>>', '<<<', '...', 64) as snippet,
                          rank
                   FROM fts_content
                   WHERE content LIKE ?
                   ORDER BY rank
                   LIMIT ?""",
                (f"%{query}%", limit),
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Embedding storage
    # ------------------------------------------------------------------

    async def store_embedding(
        self,
        source_type: str,
        source_id: int,
        chunk_text: str,
        embedding: bytes,
    ) -> None:
        now = _utcnow()
        await self.db.execute(
            """INSERT INTO embeddings (source_type, source_id, chunk_text, embedding, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_type, source_id, chunk_text, embedding, now),
        )
        await self.db.commit()

    async def get_all_embeddings(self) -> list[dict[str, Any]]:
        async with self.db.execute(
            "SELECT id, source_type, source_id, chunk_text, embedding FROM embeddings"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def clear_embeddings(self, source_type: str, source_id: int) -> None:
        await self.db.execute(
            "DELETE FROM embeddings WHERE source_type = ? AND source_id = ?",
            (source_type, source_id),
        )
        await self.db.commit()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        stats = {}
        for table in ["projects", "specs", "documents", "sessions", "embeddings"]:
            async with self.db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
                row = await cursor.fetchone()
                stats[table] = row[0] if row else 0
        return stats
