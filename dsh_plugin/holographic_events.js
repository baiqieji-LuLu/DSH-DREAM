/**
 * Pure turn-event helpers for the holographic bridge.
 *
 * Keeping these separate makes the filtering and final-assistant extraction
 * testable without booting DSH.
 */

const SYSTEM_PREFIX =
  /^\s*(Time sampled|Current runtime context|Environment:|System:)\b/i;
// DSH always emits injected blocks (runtime context, skill catalog) as a
// message that STARTS with its marker. The anchor is what keeps a real turn
// storable when it merely mentions one of these tags — for example an
// assistant explaining this very filter, or a user pasting the tag.
const INJECTED_MARKERS =
  /^\s*<(?:system-reminder|available_skills|skill_content|memory-hint)\b/i;
const ANGLE_TAG_PREFIX = /^\s*<[^>]*>/;

export function messageText(message) {
  if (!message) return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        return block?.type === "text" ? String(block.text || "") : "";
      })
      .join("");
  }
  return typeof message.text === "string" ? message.text : "";
}

export function isUsableDialogueText(role, text) {
  if (role === "system" || role === "developer") return false;
  if (typeof text !== "string" || !text.trim()) return false;
  if (SYSTEM_PREFIX.test(text)) return false;
  if (INJECTED_MARKERS.test(text)) return false;
  if (ANGLE_TAG_PREFIX.test(text)) return false;
  return true;
}

export function createTurnState() {
  return {
    turn: 0,
    user: "",
    assistant: "",
    manualCallId: "",
    manualSucceeded: false,
    flushed: false,
  };
}

export function applySessionEvent(state, event) {
  if (!state || !event || !event.type) return;
  if (event.type === "turn/start") {
    state.turn = Number(event.data?.turn) || 0;
    state.user = "";
    state.assistant = "";
    state.manualCallId = "";
    state.manualSucceeded = false;
    state.flushed = false;
    return;
  }
  if (event.type === "user/message") {
    const message = event.data;
    const role = message?.role || "user";
    const text = messageText(message);
    if (!isUsableDialogueText(role, text)) {
      // Do not clear an earlier real user message when a later injected
      // system-reminder or skill document appears in the event stream.
      return;
    }
    state.user = text.trim();
    return;
  }
  if (event.type === "assistant/message") {
    const data = event.data;
    const message = data?.message;
    const role = message?.role || "assistant";
    const text = messageText(message);
    state.turn = Number(data?.turn) || state.turn;
    if (isUsableDialogueText(role, text)) {
      state.assistant = text.trim();
    }
    return;
  }
  if (event.type === "tool/call" && event.data?.name === "memory_add_turn") {
    state.manualCallId = String(event.data.callId || "");
  }
  if (
    event.type === "tool/result" &&
    state.manualCallId &&
    String(event.data?.message?.toolCallId || "") === state.manualCallId
  ) {
    state.manualSucceeded = !event.data?.message?.isError;
  }
}

/**
 * Recover the final user/assistant text from the committed session events.
 * This is the fallback when turn-stopping arrives before our buffer contains
 * the final assistant message.
 */
export function deriveTurnFromEvents(events, turn) {
  let active = false;
  let user = "";
  let assistant = "";
  for (const event of Array.isArray(events) ? events : []) {
    if (event?.type === "turn/start") {
      active = Number(event.data?.turn) === Number(turn);
      if (active) {
        user = "";
        assistant = "";
      }
      continue;
    }
    if (!active) continue;
    if (event.type === "user/message") {
      const text = messageText(event.data);
      if (isUsableDialogueText(event.data?.role || "user", text)) {
        user = text.trim();
      }
    }
    if (event.type === "assistant/message") {
      const data = event.data;
      if (Number(data?.turn) === Number(turn)) {
        const text = messageText(data?.message);
        if (isUsableDialogueText(data?.message?.role || "assistant", text)) {
          assistant = text.trim();
        }
      }
    }
  }
  return { user, assistant };
}

export function finalizeTurn(state, events, turn) {
  let user =
    isUsableDialogueText("user", state?.user) ? state.user.trim() : "";
  let assistant =
    isUsableDialogueText("assistant", state?.assistant)
      ? state.assistant.trim()
      : "";
  if (!user || !assistant) {
    const derived = deriveTurnFromEvents(events, turn);
    user = user || derived.user;
    assistant = assistant || derived.assistant;
  }
  if (!user || !assistant) return null;
  return { user, assistant };
}
