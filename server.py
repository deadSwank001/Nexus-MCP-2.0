"""
Nexus MCP Server — the main MCP server with all 20 tools.

Uses the official Anthropic `mcp` Python SDK with stdio transport.
Each tool maps to a core domain operation with full type validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from nexus_mcp.core.project_manager import ProjectManager
from nexus_mcp.core.search_engine import SearchEngine
from nexus_mcp.core.session_manager import SessionManager
from nexus_mcp.core.spec_manager import SpecManager
from nexus_mcp.intelligence.ollama_client import OllamaClient
from nexus_mcp.storage.database import Database
from nexus_mcp.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> Tool:
    """Helper to create MCP Tool definitions cleanly."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


# ---------------------------------------------------------------------------
# Tool definitions — 20 tools organized by category
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    # ── Project Management (5) ────────────────────────────────────────
    _tool("create_project",
          "Create a new project with vision, tech stack, and summary. "
          "All context is stored in ~/.nexus/ to keep your codebase clean.",
          {"project_name": {"type": "string", "description": "Kebab-case project name (e.g. 'my-app')"},
           "vision": {"type": "string", "description": "Product vision and goals (markdown, 200+ chars)"},
           "tech_stack": {"type": "string", "description": "Technology decisions and rationale (markdown, 150+ chars)"},
           "summary": {"type": "string", "description": "Concise project summary (100+ chars)"},
           "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization"}},
          ["project_name", "vision", "tech_stack", "summary"]),

    _tool("load_project",
          "Load complete project context. With summarize=true, uses Ollama to create "
          "a condensed version optimized for LLM token windows.",
          {"project_name": {"type": "string", "description": "Name of the project to load"},
           "summarize": {"type": "boolean", "description": "Use AI to summarize for token efficiency", "default": False},
           "token_budget": {"type": "integer", "description": "Target token count for summarized output", "default": 0}},
          ["project_name"]),

    _tool("list_projects",
          "List all projects with metadata, spec counts, and summaries.",
          {"include_archived": {"type": "boolean", "description": "Include archived projects", "default": False}},
          []),

    _tool("analyze_codebase",
          "Analyze an existing codebase and create/update a project with AI-generated "
          "vision, tech stack, and summary based on the file structure.",
          {"project_name": {"type": "string", "description": "Name for the project"},
           "file_list": {"type": "string", "description": "Newline-separated list of files in the codebase"},
           "sample_content": {"type": "string", "description": "Sample code/config content for analysis"}},
          ["project_name", "file_list"]),

    _tool("archive_project",
          "Soft-delete a project. Can be restored by listing with include_archived=true.",
          {"project_name": {"type": "string", "description": "Name of the project to archive"}},
          ["project_name"]),

    # ── Specification Management (5) ──────────────────────────────────
    _tool("create_spec",
          "Create a timestamped feature specification with optional AI-suggested task breakdown.",
          {"project_name": {"type": "string", "description": "Project to create spec in"},
           "feature_name": {"type": "string", "description": "Human-readable feature name"},
           "spec_content": {"type": "string", "description": "Feature specification (markdown)"},
           "tasks_content": {"type": "string", "description": "Task list (markdown checkboxes)"},
           "notes_content": {"type": "string", "description": "Design notes (markdown)"},
           "auto_suggest_tasks": {"type": "boolean", "description": "Use AI to suggest tasks from spec content"}},
          ["project_name", "feature_name"]),

    _tool("load_spec",
          "Load a specification with full content and project context. "
          "Supports fuzzy matching on spec names.",
          {"project_name": {"type": "string", "description": "Project containing the spec"},
           "spec_name": {"type": "string", "description": "Spec name (supports fuzzy matching)"}},
          ["project_name", "spec_name"]),

    _tool("update_spec",
          "Update spec content directly. Unlike foundry-mcp's complex edit command system, "
          "Nexus uses simple direct content replacement — the AI agent handles text manipulation.",
          {"project_name": {"type": "string", "description": "Project containing the spec"},
           "spec_name": {"type": "string", "description": "Name of the spec to update"},
           "spec_content": {"type": "string", "description": "New spec content (null to skip)"},
           "tasks_content": {"type": "string", "description": "New tasks content (null to skip)"},
           "notes_content": {"type": "string", "description": "New notes content (null to skip)"},
           "status": {"type": "string", "enum": ["active", "completed", "on_hold", "cancelled"]}},
          ["project_name", "spec_name"]),

    _tool("delete_spec",
          "Delete a specification with confirmation.",
          {"project_name": {"type": "string", "description": "Project containing the spec"},
           "spec_name": {"type": "string", "description": "Name of the spec to delete"},
           "confirm": {"type": "boolean", "description": "Must be true to confirm deletion"}},
          ["project_name", "spec_name"]),

    _tool("diff_spec",
          "Preview changes before updating a spec.",
          {"project_name": {"type": "string", "description": "Project containing the spec"},
           "spec_name": {"type": "string", "description": "Name of the spec"},
           "new_spec_content": {"type": "string", "description": "Proposed spec content"},
           "new_tasks_content": {"type": "string", "description": "Proposed tasks content"},
           "new_notes_content": {"type": "string", "description": "Proposed notes content"}},
          ["project_name", "spec_name"]),

    # ── Intelligence Tools (5) — NEW ─────────────────────────────────
    _tool("search",
          "Semantic search across ALL projects using hybrid keyword + vector search. "
          "Finds relevant context even with different wording.",
          {"query": {"type": "string", "description": "Search query (natural language)"},
           "top_k": {"type": "integer", "description": "Max results to return", "default": 10},
           "project_filter": {"type": "string", "description": "Optional: limit to one project"}},
          ["query"]),

    _tool("suggest_next_steps",
          "AI-powered analysis of current project state with actionable recommendations.",
          {"project_name": {"type": "string", "description": "Project to analyze"},
           "current_state": {"type": "string", "description": "Description of what you're currently working on"}},
          ["project_name"]),

    _tool("find_related",
          "Find content related to a project or spec across ALL projects. "
          "Discovers cross-project patterns and shared concepts.",
          {"project_name": {"type": "string", "description": "Source project"},
           "spec_name": {"type": "string", "description": "Optional: specific spec to find relations for"}},
          ["project_name"]),

    _tool("ask_context",
          "Ask a natural language question about your projects and get an AI-generated answer "
          "based on stored knowledge.",
          {"question": {"type": "string", "description": "Your question in natural language"},
           "project_name": {"type": "string", "description": "Optional: limit context to one project"}},
          ["question"]),

    _tool("summarize_context",
          "Generate a condensed summary of project context, optimized for LLM token windows.",
          {"project_name": {"type": "string", "description": "Project to summarize"},
           "max_length": {"type": "integer", "description": "Maximum summary length in characters", "default": 1000}},
          ["project_name"]),

    # ── Session Management (3) — NEW ─────────────────────────────────
    _tool("start_session",
          "Start a tracked development session. Shows context from previous sessions.",
          {"project_name": {"type": "string", "description": "Project to work on"}},
          ["project_name"]),

    _tool("end_session",
          "End the current session with a summary of what was accomplished.",
          {"session_id": {"type": "integer", "description": "Session ID to end"},
           "summary": {"type": "string", "description": "What was accomplished"},
           "decisions": {"type": "array", "items": {"type": "string"}, "description": "Key decisions made"}},
          ["session_id"]),

    _tool("resume_session",
          "Resume work on a project. Loads previous session context and starts a new session.",
          {"project_name": {"type": "string", "description": "Project to resume work on"}},
          ["project_name"]),

    # ── System (2) ───────────────────────────────────────────────────
    _tool("nexus_status",
          "System health check — Ollama connectivity, database stats, storage info.",
          {}, []),

    _tool("nexus_help",
          "Get contextual help and workflow examples.",
          {"topic": {"type": "string", "description": "Help topic: 'getting_started', 'projects', 'specs', 'search', 'sessions', 'all'", "default": "getting_started"}},
          []),
]


class NexusMcpServer:
    """Main MCP server orchestrating all components."""

    def __init__(self):
        self._db = Database()
        self._ollama = OllamaClient()
        self._vectors = VectorStore(self._db)
        self._projects = ProjectManager(self._db, self._ollama, self._vectors)
        self._specs = SpecManager(self._db, self._ollama, self._vectors)
        self._sessions = SessionManager(self._db, self._ollama)
        self._search = SearchEngine(self._db, self._ollama, self._vectors)

    async def run(self) -> None:
        """Start the MCP server with stdio transport."""
        server = Server("nexus-mcp")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return TOOLS

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
            args = arguments or {}
            try:
                result = await self._handle_tool(name, args)
                return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
            except Exception as e:
                error_response = {"error": str(e), "tool": name}
                return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

        # Initialize database
        await self._db.connect()

        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())
        finally:
            await self._ollama.close()
            await self._db.close()

    async def _handle_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route tool calls to domain operations."""
        # ── Project tools ──
        if name == "create_project":
            return await self._projects.create_project(
                name=args["project_name"],
                vision=args.get("vision", ""),
                tech_stack=args.get("tech_stack", ""),
                summary=args.get("summary", ""),
                tags=args.get("tags"),
            )
        elif name == "load_project":
            return await self._projects.load_project(
                name=args["project_name"],
                summarize=args.get("summarize", False),
                token_budget=args.get("token_budget", 0),
            )
        elif name == "list_projects":
            return await self._projects.list_projects(
                include_archived=args.get("include_archived", False),
            )
        elif name == "analyze_codebase":
            return await self._projects.analyze_codebase(
                name=args["project_name"],
                file_list=args["file_list"],
                sample_content=args.get("sample_content", ""),
            )
        elif name == "archive_project":
            return await self._projects.archive_project(name=args["project_name"])

        # ── Spec tools ──
        elif name == "create_spec":
            return await self._specs.create_spec(
                project_name=args["project_name"],
                feature_name=args["feature_name"],
                spec_content=args.get("spec_content", ""),
                tasks_content=args.get("tasks_content", ""),
                notes_content=args.get("notes_content", ""),
                auto_suggest_tasks=args.get("auto_suggest_tasks", False),
            )
        elif name == "load_spec":
            return await self._specs.load_spec(
                project_name=args["project_name"],
                spec_name=args["spec_name"],
            )
        elif name == "update_spec":
            return await self._specs.update_spec(
                project_name=args["project_name"],
                spec_name=args["spec_name"],
                spec_content=args.get("spec_content"),
                tasks_content=args.get("tasks_content"),
                notes_content=args.get("notes_content"),
                status=args.get("status"),
            )
        elif name == "delete_spec":
            return await self._specs.delete_spec(
                project_name=args["project_name"],
                spec_name=args["spec_name"],
                confirm=args.get("confirm", False),
            )
        elif name == "diff_spec":
            return await self._specs.diff_spec(
                project_name=args["project_name"],
                spec_name=args["spec_name"],
                new_spec_content=args.get("new_spec_content"),
                new_tasks_content=args.get("new_tasks_content"),
                new_notes_content=args.get("new_notes_content"),
            )

        # ── Intelligence tools ──
        elif name == "search":
            return await self._search.search(
                query=args["query"],
                top_k=args.get("top_k", 10),
                project_filter=args.get("project_filter"),
            )
        elif name == "suggest_next_steps":
            project = await self._projects.load_project(args["project_name"])
            context = json.dumps(project, indent=2, default=str)
            steps = await self._ollama.suggest_next_steps(context, args.get("current_state", ""))
            return {"project_name": args["project_name"], "suggestions": steps}
        elif name == "find_related":
            return await self._search.find_related(
                project_name=args["project_name"],
                spec_name=args.get("spec_name"),
            )
        elif name == "ask_context":
            return await self._search.ask(
                question=args["question"],
                project_name=args.get("project_name"),
            )
        elif name == "summarize_context":
            project = await self._projects.load_project(args["project_name"])
            context = json.dumps(project, indent=2, default=str)
            summary = await self._ollama.summarize(context, args.get("max_length", 1000))
            return {"project_name": args["project_name"], "summary": summary}

        # ── Session tools ──
        elif name == "start_session":
            return await self._sessions.start_session(args["project_name"])
        elif name == "end_session":
            return await self._sessions.end_session(
                session_id=args["session_id"],
                summary=args.get("summary", ""),
                decisions=args.get("decisions"),
            )
        elif name == "resume_session":
            return await self._sessions.resume_session(args["project_name"])

        # ── System tools ──
        elif name == "nexus_status":
            return await self._get_status()
        elif name == "nexus_help":
            return self._get_help(args.get("topic", "getting_started"))

        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _get_status(self) -> dict[str, Any]:
        db_stats = await self._db.get_stats()
        ollama_health = await self._ollama.health_check()
        vec_count = await self._vectors.count()
        return {
            "version": "0.1.0",
            "database": db_stats,
            "ollama": ollama_health,
            "embeddings_count": vec_count,
            "storage_path": str(Database._db_path if hasattr(Database, '_db_path') else "~/.nexus/"),
        }

    def _get_help(self, topic: str) -> dict[str, Any]:
        help_content = {
            "getting_started": {
                "title": "Getting Started with Nexus MCP",
                "steps": [
                    "1. Create a project: create_project with name, vision, tech_stack, summary",
                    "2. Create specs: create_spec with feature details and task breakdowns",
                    "3. Load context: load_project to resume work on a project",
                    "4. Search: use search to find anything across all projects",
                    "5. Track sessions: start_session/end_session for continuity",
                ],
                "tip": "Nexus works best with Ollama running locally for AI features. "
                       "Without Ollama, all CRUD operations still work perfectly.",
            },
            "projects": {
                "title": "Project Management",
                "tools": ["create_project", "load_project", "list_projects", "analyze_codebase", "archive_project"],
                "workflow": "Create → Load → Work → Update. Use analyze_codebase for existing repos.",
            },
            "specs": {
                "title": "Specification Management",
                "tools": ["create_spec", "load_spec", "update_spec", "delete_spec", "diff_spec"],
                "workflow": "Create spec → Work on tasks → Update progress → Mark complete.",
            },
            "search": {
                "title": "Search & Intelligence",
                "tools": ["search", "find_related", "ask_context", "suggest_next_steps", "summarize_context"],
                "tip": "search uses hybrid keyword + semantic search. ask_context lets you query in natural language.",
            },
            "sessions": {
                "title": "Session Management",
                "tools": ["start_session", "end_session", "resume_session"],
                "workflow": "start_session → work → end_session with summary. resume_session loads previous context.",
            },
        }
        if topic == "all":
            return {"topics": help_content}
        return help_content.get(topic, help_content["getting_started"])
