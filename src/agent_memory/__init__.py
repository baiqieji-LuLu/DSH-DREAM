"""Layered, auditable long-term memory for LLM agents.

``agent_memory`` separates memory concerns into small, testable layers:

* :class:`AnchorManager` owns immutable, versioned hard constraints.
* :class:`WindowManager` fits recent conversation turns into a token budget.
* :class:`RAGStore` stores only explicitly trusted memory sources.
* :class:`AuditLogger` records what context was assembled for each request.
* :class:`MemoryAgent` wraps the store with optional LLM extraction, rewriting,
  reranking, and generation.
* :func:`render_prompt` turns those pieces into an injection-resistant prompt.
* :func:`memory_search`, :func:`memory_add`, and :func:`get_rendered_context`
  are the tool-calling entry points exposed to an agent runtime.

The public names are declared here and resolved lazily, so importing the
package does not load Chroma or a sentence-transformer model. Heavy
dependencies are imported only when their component is first used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .anchor_manager import AnchorManager
    from .audit_logger import AuditLogger
    from .memory_agent import (
        MemoryAgent,
        MemoryAgentResult,
        build_deepseek_client,
        filter_historical_questions,
    )
    from .config import MemoryConfig, load_config
    from .plugin_entry import (
        get_plugin_status,
        get_rendered_context,
        memory_add,
        memory_search,
    )
    from .rag_store import RAGStore
    from .renderer import render_prompt
    from .window_manager import WindowManager

__version__: Final = "0.3.6"

#: Mapping of public name -> (implementation module, attribute name).
_LAZY_IMPORTS: Final[dict[str, tuple[str, str]]] = {
    "AnchorManager": ("anchor_manager", "AnchorManager"),
    "WindowManager": ("window_manager", "WindowManager"),
    "RAGStore": ("rag_store", "RAGStore"),
    "AuditLogger": ("audit_logger", "AuditLogger"),
    "MemoryAgent": ("memory_agent", "MemoryAgent"),
    "MemoryAgentResult": ("memory_agent", "MemoryAgentResult"),
    "build_deepseek_client": ("memory_agent", "build_deepseek_client"),
    "filter_historical_questions": ("memory_agent", "filter_historical_questions"),
    "MemoryConfig": ("config", "MemoryConfig"),
    "load_config": ("config", "load_config"),
    "get_plugin_status": ("plugin_entry", "get_plugin_status"),
    "render_prompt": ("renderer", "render_prompt"),
    "memory_search": ("plugin_entry", "memory_search"),
    "memory_add": ("plugin_entry", "memory_add"),
    "memory_add_turn": ("plugin_entry", "memory_add_turn"),
    "get_rendered_context": ("plugin_entry", "get_rendered_context"),
}

__all__: Final[tuple[str, ...]] = (
    "AnchorManager",
    "AuditLogger",
    "MemoryAgent",
    "MemoryAgentResult",
    "build_deepseek_client",
    "filter_historical_questions",
    "MemoryConfig",
    "load_config",
    "get_plugin_status",
    "RAGStore",
    "WindowManager",
    "get_rendered_context",
    "memory_add",
    "memory_add_turn",
    "memory_search",
    "render_prompt",
)


def __getattr__(name: str) -> object:
    """Resolve a public API name on first access.

    Args:
        name: Attribute requested from the package.

    Returns:
        The object exported by the owning implementation module.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
        ImportError: If the owning module is absent from this installation.
    """
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    qualified_module = f"{__name__}.{module_name}"
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
    except ModuleNotFoundError as exc:
        if exc.name == qualified_module:
            raise ImportError(
                f"{name!r} is part of the public API, but its implementation module "
                f"{qualified_module!r} is not available in this build."
            ) from exc
        raise

    value: object = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the names exposed by the package, including lazy exports."""
    return sorted({*globals(), *__all__})
