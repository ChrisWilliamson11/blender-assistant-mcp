"""RAG tools exposed to the internal MCP registry.

These tools let the LLM explicitly query the bundled Blender docs RAG
and inspect basic status, independent of automatic augmentation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..rag_system import ensure_rag_loaded_async, get_rag_instance, is_rag_enabled


def _to_set_from_any(val: Any) -> Optional[set]:
    """Coerce various inputs into a set of strings.
    Accepts list/tuple/set or comma-separated string. Returns None if empty/invalid.
    """
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        items = [str(x).strip() for x in val if str(x).strip()]
        return set(items) if items else None
    if isinstance(val, str):
        items = [s.strip() for s in val.split(",") if s.strip()]
        return set(items) if items else None
    return None


def rag_query(
    query: str,
    num_results: int = 5,
    prefer_source: Optional[str] = None,
    page_types: Any = None,
    excerpt_chars: int = 600,
) -> Dict[str, Any]:
    """Query the bundled Blender docs RAG and return top matches with metadata and excerpts.

    Args:
        query: Search query or question
        num_results: Number of results to return (default 5)
        prefer_source: Optional bias, one of: "API" | "Manual"
        page_types: Optional filter list or comma-separated string of page types
        excerpt_chars: Max characters to include from each content (default 600)
    """
    try:
        # Kick off background load if needed; returns immediately
        ensure_rag_loaded_async()
        rag = get_rag_instance()
        if not rag.enabled:
            return {
                "success": False,
                "enabled": False,
                "message": "RAG is not enabled or database not found (still loading?). Build it or bundle rag_db.",
            }

        # Normalize prefer_source
        pref = None
        if isinstance(prefer_source, str) and prefer_source.strip():
            ps = prefer_source.strip().lower()
            if ps in ("api", "manual"):
                pref = "API" if ps == "api" else "Manual"
            else:
                pref = None

        # Query
        docs = (
            rag.query_documents(
                query=query,
                n_results=max(1, int(num_results or 5)),
                prefer_source=pref,
                exclude_indices=None,
            )
            or []
        )

        # Filter by page_types if provided
        allowed_types = _to_set_from_any(page_types)
        if allowed_types:
            docs = [d for d in docs if str(d.get("metadata", {}).get("page_type", "")) in allowed_types]

        # Build response
        results: List[Dict[str, Any]] = []
        max_len = max(0, int(excerpt_chars or 0))
        for d in docs:
            md = d.get("metadata", {}) or {}
            text = d.get("content", "") or ""
            if max_len and len(text) > max_len:
                excerpt = text[:max_len - 1] + "\u2026"
            else:
                excerpt = text
            results.append(
                {
                    "content": excerpt,
                    "similarity": float(d.get("similarity", 0.0)),
                    "source": md.get("source"),
                    "page_type": md.get("page_type"),
                    "file": md.get("file"),
                    "url": md.get("url"),
                    "index": d.get("index"),
                }
            )

        return {
            "success": True,
            "enabled": True,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        return {"success": False, "error": f"rag_query failed: {str(e)}"}


def rag_get_stats() -> Dict[str, Any]:
    """Return basic RAG status and stats (enabled, document_count, db_path)."""
    try:
        ensure_rag_loaded_async()
        rag = get_rag_instance()
        return rag.get_stats()
    except Exception as e:
        return {"enabled": False, "error": f"rag_get_stats failed: {str(e)}"}


def register():
    """Register RAG tools with the MCP registry."""
    from . import tool_registry

    tool_registry.register_tool(
        "rag_query",
        rag_query,
        "Query the bundled Blender docs RAG and return top matches with metadata and excerpts. Supports source bias and page_type filtering.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query or question"},
                "num_results": {"type": "integer", "description": "Number of results (default 5)"},
                "prefer_source": {"type": "string", "description": "Optional bias: 'API' or 'Manual'"},
                "page_types": {"type": ["array", "string"], "description": "Filter by page types (list or comma-separated)"},
                "excerpt_chars": {"type": "integer", "description": "Max characters for each excerpt (default 600)"},
            },
            "required": ["query"],
        },
        category="RAG",
    )

    tool_registry.register_tool(
        "rag_get_stats",
        rag_get_stats,
        "Get RAG status and basic stats (enabled, document_count, db_path).",
        {"type": "object", "properties": {}},
        category="RAG",
    )
