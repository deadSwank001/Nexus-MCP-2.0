"""MCP server layer — tool definitions and server lifecycle."""
from nexus_mcp.mcp.server import NexusMcpServer

__all__ = ["NexusMcpServer"]

#2
"""
Nexus MCP — Ollama-powered intelligent project context management.

An MCP server that understands your projects, connects concepts across codebases,
and proactively surfaces relevant context. Built as a superior replacement for foundry-mcp.
"""

__version__ = "0.1.0"
__author__ = "Nexus MCP"


#3
"""Intelligence layer — Ollama-powered AI capabilities."""
from nexus_mcp.intelligence.ollama_client import OllamaClient

__all__ = ["OllamaClient"]


#4
"""Core domain logic — project, spec, and session management."""
from nexus_mcp.core.project_manager import ProjectManager
from nexus_mcp.core.spec_manager import SpecManager
from nexus_mcp.core.session_manager import SessionManager
from nexus_mcp.core.search_engine import SearchEngine

__all__ = ["ProjectManager", "SpecManager", "SessionManager", "SearchEngine"]


#5
