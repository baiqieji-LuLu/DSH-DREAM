import assert from "node:assert/strict";
import {
  applySessionEvent,
  createTurnState,
  deriveTurnFromEvents,
  finalizeTurn,
  isUsableDialogueText,
} from "../dsh_plugin/holographic_events.js";

const SYSTEM_TIME =
  "Time sampled while preparing turn 1: 2026-10-27T23:29:19+08:00[Asia/Shanghai]";

function userEvent(text, role = "user") {
  return {
    type: "user/message",
    data: { role, content: [{ type: "text", text }] },
  };
}

function assistantEvent(turn, text) {
  return {
    type: "assistant/message",
    data: {
      turn,
      message: {
        role: "assistant",
        content: [
          { type: "reasoning", text: "internal" },
          { type: "text", text },
        ],
      },
    },
  };
}

// 1. System time is rejected even if DSH labels it as a user message.
assert.equal(isUsableDialogueText("user", SYSTEM_TIME), false);
const dirty = createTurnState();
applySessionEvent(dirty, { type: "turn/start", data: { turn: 1 } });
applySessionEvent(dirty, userEvent(SYSTEM_TIME, "user"));
applySessionEvent(dirty, assistantEvent(1, "should not pair with system time"));
assert.equal(dirty.user, "");
assert.equal(finalizeTurn(dirty, [], 1), null);

// 2. Explicit system/developer roles are rejected.
assert.equal(isUsableDialogueText("system", "anything"), false);
assert.equal(isUsableDialogueText("developer", "anything"), false);

// 3. Real user + assistant are captured and timestamped in one turn.
const clean = createTurnState();
applySessionEvent(clean, { type: "turn/start", data: { turn: 2 } });
applySessionEvent(clean, userEvent("I had hotpot tonight."));
applySessionEvent(clean, assistantEvent(2, "That sounds delicious!"));
const cleanTurn = finalizeTurn(clean, [], 2);
assert.deepEqual(cleanTurn, {
  user: "I had hotpot tonight.",
  assistant: "That sounds delicious!",
});

// 4. turn-stopping fallback: assistant event was not in the incremental buffer.
const fallback = createTurnState();
applySessionEvent(fallback, { type: "turn/start", data: { turn: 3 } });
applySessionEvent(fallback, userEvent("What should I drink?"));
const committedEvents = [
  { type: "turn/start", data: { turn: 3 } },
  userEvent("What should I drink?"),
  assistantEvent(3, "Try an iced americano."),
];
assert.deepEqual(finalizeTurn(fallback, committedEvents, 3), {
  user: "What should I drink?",
  assistant: "Try an iced americano.",
});

// 5. The committed-event fallback itself filters injected system time.
assert.deepEqual(
  deriveTurnFromEvents(
    [
      { type: "turn/start", data: { turn: 4 } },
      userEvent(SYSTEM_TIME),
      assistantEvent(4, "This should not be written."),
    ],
    4,
  ),
  { user: "", assistant: "This should not be written." },
);

// 6. Injected payloads are rejected when they START the message — that is the
//    only position DSH uses for them.
for (const injected of [
  "<system-reminder>hidden instruction</system-reminder>",
  "<available_skills>skill catalog</available_skills>",
  "<skill_content>body</skill_content>",
  "<memory-hint>retrieved memory</memory-hint>",
  "<div>angle tag prefix</div>",
]) {
  assert.equal(isUsableDialogueText("user", injected), false, injected);
}

// 6b. Regression (session-24b2d960 turn 1): a real turn that merely MENTIONS a
//     marker mid-text was dropped at turn end, so nothing was ever written.
//     Both the user and the assistant side must survive.
const mentioned = "消息里含有 <system-reminder> 字样时不应整段丢弃";
assert.equal(isUsableDialogueText("user", mentioned), true);
const quoted = createTurnState();
applySessionEvent(quoted, { type: "turn/start", data: { turn: 8 } });
applySessionEvent(quoted, userEvent("确认一下全息过滤逻辑"));
applySessionEvent(quoted, assistantEvent(8, mentioned));
assert.deepEqual(finalizeTurn(quoted, [], 8), {
  user: "确认一下全息过滤逻辑",
  assistant: mentioned,
});

// 7. A late system-reminder must not erase the real user message.
const preserved = createTurnState();
applySessionEvent(preserved, { type: "turn/start", data: { turn: 6 } });
applySessionEvent(preserved, userEvent("今天天气不错，出去散步吧。"));
applySessionEvent(
  preserved,
  userEvent("<system-reminder>diagnose-windows-sandbox-acl</system-reminder>"),
);
applySessionEvent(preserved, assistantEvent(6, "好呀，散步很舒服。"));
assert.deepEqual(finalizeTurn(preserved, [], 6), {
  user: "今天天气不错，出去散步吧。",
  assistant: "好呀，散步很舒服。",
});

// 8. The committed-event fallback also ignores a later injected reminder.
assert.deepEqual(
  deriveTurnFromEvents(
    [
      { type: "turn/start", data: { turn: 7 } },
      userEvent("今天天气不错。"),
      userEvent("<available_skills>tools</available_skills>"),
      assistantEvent(7, "适合散步。"),
    ],
    7,
  ),
  { user: "今天天气不错。", assistant: "适合散步。" },
);

// 9. A successful memory_add_turn tool call marks the turn as manually handled.
const manual = createTurnState();
applySessionEvent(manual, { type: "turn/start", data: { turn: 5 } });
applySessionEvent(manual, {
  type: "tool/call",
  data: { name: "memory_add_turn", callId: "call-1" },
});
applySessionEvent(manual, {
  type: "tool/result",
  data: { message: { toolCallId: "call-1", isError: false } },
});
assert.equal(manual.manualSucceeded, true);

console.log("holographic_events: PASS");
