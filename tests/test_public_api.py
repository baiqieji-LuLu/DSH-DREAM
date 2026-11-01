"""Contract tests for the package's public API.

These tests intentionally avoid importing the heavy implementation modules so
they stay fast and independent of Chroma or model availability.
"""

from __future__ import annotations

import unittest

import agent_memory

EXPECTED_EXPORTS = {
    "AnchorManager",
    "WindowManager",
    "RAGStore",
    "MemoryAgent",
    "MemoryAgentResult",
    "build_deepseek_client",
    "filter_historical_questions",
    "MemoryConfig",
    "load_config",
    "get_plugin_status",
    "AuditLogger",
    "render_prompt",
    "memory_search",
    "memory_add",
    "memory_add_turn",
    "get_rendered_context",
}


class PublicApiTests(unittest.TestCase):
    def test_version_is_a_non_empty_string(self) -> None:
        self.assertIsInstance(agent_memory.__version__, str)
        self.assertRegex(agent_memory.__version__, r"^\d+\.\d+\.\d+")

    def test_all_declares_the_expected_exports(self) -> None:
        self.assertEqual(set(agent_memory.__all__), EXPECTED_EXPORTS)

    def test_public_names_are_unique(self) -> None:
        self.assertEqual(len(agent_memory.__all__), len(set(agent_memory.__all__)))

    def test_dir_includes_lazy_exports(self) -> None:
        self.assertTrue(EXPECTED_EXPORTS.issubset(dir(agent_memory)))

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            _ = agent_memory.not_a_public_name


if __name__ == "__main__":
    unittest.main()
