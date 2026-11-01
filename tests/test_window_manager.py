"""Unit tests for budgeted conversation-window trimming."""

from __future__ import annotations

import unittest

from agent_memory.window_manager import (
    Message,
    WindowBudgetError,
    WindowManager,
    WindowValidationError,
)


def word_count(text: str) -> int:
    """Small deterministic token counter used by the tests."""
    return len(text.split())


class WindowManagerTests(unittest.TestCase):
    def make_manager(self, max_tokens: int) -> WindowManager:
        return WindowManager(
            max_tokens=max_tokens,
            token_counter=word_count,
            message_overhead=0,
        )

    def test_latest_turn_is_protected_and_older_turns_are_dropped(self) -> None:
        manager = self.make_manager(6)
        manager.add_message("user", "old question")
        manager.add_message("assistant", "old answer")
        manager.add_message("user", "recent question")
        manager.add_message("assistant", "recent answer")

        window = manager.build_window()
        self.assertEqual(
            [message.content for message in window],
            ["recent question", "recent answer"],
        )
        self.assertEqual(manager.window_tokens(), 6)

    def test_protected_turn_is_never_partially_truncated(self) -> None:
        manager = self.make_manager(5)
        manager.add_message("user", "recent question")
        manager.add_message("assistant", "recent answer")

        with self.assertRaises(WindowBudgetError) as raised:
            manager.build_window()
        self.assertEqual(raised.exception.required_tokens, 6)
        self.assertEqual(raised.exception.max_tokens, 5)
        self.assertEqual(len(manager.history), 2)

    def test_trimming_keeps_a_contiguous_suffix(self) -> None:
        manager = self.make_manager(12)
        manager.add_message("user", "one two three four")
        manager.add_message("assistant", "five six")
        manager.add_message("user", "seven eight")
        manager.add_message("assistant", "nine ten")

        window = manager.build_window()
        self.assertEqual(
            [message.content for message in window],
            ["five six", "seven eight", "nine ten"],
        )

    def test_mapping_messages_and_explicit_history_are_supported(self) -> None:
        manager = self.make_manager(100)
        window = manager.build_window(
            [
                {"role": "user", "content": "stored question"},
                {"role": "assistant", "content": "stored answer", "name": "helper"},
            ]
        )
        self.assertEqual(len(window), 2)
        self.assertIsInstance(window[0], Message)
        self.assertEqual(window[1].name, "helper")
        self.assertEqual(len(manager.history), 0)

    def test_without_a_user_message_the_final_message_is_protected(self) -> None:
        manager = self.make_manager(3)
        manager.add_message("assistant", "older reply")
        manager.add_message("assistant", "newest reply")
        self.assertEqual(
            [message.content for message in manager.build_window()],
            ["newest reply"],
        )

    def test_window_hash_is_stable_and_content_sensitive(self) -> None:
        manager = self.make_manager(100)
        manager.add_message("user", "hello")
        manager.add_message("assistant", "hi")
        first = manager.window_hash()
        second = manager.window_hash()
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

        manager.add_message("user", "changed")
        self.assertNotEqual(first, manager.window_hash())

    def test_extend_is_atomic_when_a_message_is_invalid(self) -> None:
        manager = self.make_manager(100)
        manager.add_message("user", "kept")
        with self.assertRaises(WindowValidationError):
            manager.extend(
                [
                    {"role": "assistant", "content": "valid"},
                    {"role": "user"},
                ]
            )
        self.assertEqual([message.content for message in manager.history], ["kept"])

    def test_invalid_configuration_and_messages_are_rejected(self) -> None:
        with self.assertRaises(WindowValidationError):
            self.make_manager(0)
        with self.assertRaises(WindowValidationError):
            Message(role="invalid", content="hello")
        with self.assertRaises(WindowValidationError):
            Message(role="user", content="   ")
        with self.assertRaises(WindowValidationError):
            self.make_manager(100).extend("not a message list")


if __name__ == "__main__":
    unittest.main()
