"""
Tests for Nexus MCP core functionality.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexus_mcp.storage.database import Database
from nexus_mcp.storage.vector_store import VectorStore
from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.core.project_manager import ProjectManager
from nexus_mcp.core.spec_manager import SpecManager, _generate_spec_name
from nexus_mcp.core.session_manager import SessionManager
from nexus_mcp.core.search_engine import SearchEngine


def run(coro):
    """Helper to run async functions in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDatabase:
    """Test the SQLite database layer."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=Path(self.tmp.name))
        run(self.db.connect())

    def teardown_method(self):
        run(self.db.close())
        os.unlink(self.tmp.name)

    def test_create_project(self):
        result = run(self.db.create_project(
            name="test-project",
            vision="Test vision",
            tech_stack="Python, SQLite",
            summary="A test project",
        ))
        assert result is not None
        assert result["name"] == "test-project"
        assert result["vision"] == "Test vision"

    def test_duplicate_project_fails(self):
        run(self.db.create_project(name="dup-test"))
        try:
            run(self.db.create_project(name="dup-test"))
            assert False, "Should have raised"
        except Exception:
            pass  # Expected

    def test_list_projects(self):
        run(self.db.create_project(name="proj-1"))
        run(self.db.create_project(name="proj-2"))
        projects = run(self.db.list_projects())
        assert len(projects) == 2

    def test_archive_project(self):
        run(self.db.create_project(name="to-archive"))
        success = run(self.db.archive_project("to-archive"))
        assert success
        # Should not appear in default listing
        projects = run(self.db.list_projects())
        assert len(projects) == 0
        # Should appear with include_archived
        projects = run(self.db.list_projects(include_archived=True))
        assert len(projects) == 1

    def test_create_and_load_spec(self):
        run(self.db.create_project(name="spec-test"))
        spec = run(self.db.create_spec(
            project_name="spec-test",
            name="20250101_120000_auth",
            feature_name="auth",
            spec_content="# Auth spec",
            tasks_content="- [ ] Login\n- [ ] Register",
        ))
        assert spec is not None
        assert spec["feature_name"] == "auth"

        loaded = run(self.db.get_spec("spec-test", "20250101_120000_auth"))
        assert loaded["spec_content"] == "# Auth spec"

    def test_list_specs(self):
        run(self.db.create_project(name="multi-spec"))
        run(self.db.create_spec("multi-spec", "spec_1", "feat1"))
        run(self.db.create_spec("multi-spec", "spec_2", "feat2"))
        specs = run(self.db.list_specs("multi-spec"))
        assert len(specs) == 2

    def test_delete_spec(self):
        run(self.db.create_project(name="del-spec"))
        run(self.db.create_spec("del-spec", "to_delete", "feature"))
        success = run(self.db.delete_spec("del-spec", "to_delete"))
        assert success
        loaded = run(self.db.get_spec("del-spec", "to_delete"))
        assert loaded is None

    def test_session_lifecycle(self):
        run(self.db.create_project(name="session-test"))
        session = run(self.db.start_session("session-test"))
        assert session is not None
        assert session["status"] == "active"

        success = run(self.db.end_session(
            session["id"], summary="Did stuff", decisions=["Use React"]
        ))
        assert success

    def test_fts_search(self):
        run(self.db.create_project(name="search-test"))
        run(self.db.create_spec(
            "search-test", "auth_spec", "authentication",
            spec_content="Implement OAuth2 authentication with JWT tokens",
        ))
        results = run(self.db.search_fts("OAuth2"))
        assert len(results) > 0

    def test_stats(self):
        run(self.db.create_project(name="stats-test"))
        stats = run(self.db.get_stats())
        assert stats["projects"] >= 1


class TestVectorStore:
    """Test the vector store."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = Database(db_path=Path(self.tmp.name))
        run(self.db.connect())
        self.vectors = VectorStore(self.db)

    def teardown_method(self):
        run(self.db.close())
        os.unlink(self.tmp.name)

    def test_store_and_search(self):
        # Store some embeddings
        run(self.vectors.store("project", 1, "Python web framework", [0.1, 0.2, 0.3, 0.4]))
        run(self.vectors.store("project", 2, "Machine learning pipeline", [0.5, 0.6, 0.7, 0.8]))

        # Search with a similar vector
        results = run(self.vectors.search([0.1, 0.2, 0.3, 0.4], top_k=5, min_similarity=0.0))
        assert len(results) > 0
        assert results[0]["source_type"] == "project"

    def test_clear_embeddings(self):
        run(self.vectors.store("spec", 1, "test content", [0.1, 0.2, 0.3]))
        run(self.vectors.clear("spec", 1))
        count = run(self.vectors.count())
        assert count == 0


class TestSpecManager:
    """Test spec name generation."""

    def test_generate_spec_name(self):
        name = _generate_spec_name("user authentication")
        assert "user_authentication" in name
        # Should have timestamp prefix
        parts = name.split("_")
        assert len(parts[0]) == 8  # YYYYMMDD

    def test_generate_spec_name_special_chars(self):
        name = _generate_spec_name("OAuth2 Login!")
        assert "oauth2" in name.lower()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
