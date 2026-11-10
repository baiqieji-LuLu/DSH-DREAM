/**
 * Top-3 automatic memory retrieval injected at agent/pre-step.
 *
 * v0.4.0: the decision "does this turn need memory?" belongs to the
 * RouterEngine; this module only performs retrieval when the router says
 * `retrieve`.  Two escape hatches are mandatory:
 *
 * - `router.enabled === false` -> the exact legacy behaviour (length/greeting
 *   gate, then always retrieve), so a bad rollout can be reverted in place;
 * - `router.shadow_mode === true` -> retrieves exactly like the legacy path but
 *   only records what the router would have decided.
 *
 * The router never influences what gets injected: the `0.6` top-1 threshold and
 * `createMemoryHint` are unchanged, and an explicit `memory_search` tool call
 * never goes through here.
 */
import { isUsableDialogueText, messageText } from "./holographic_events.js";
import { hashQuery } from "./router_engine.js";

// `@deepseek-ai/dsh-llm` is a Host peer dependency. Keeping the import lazy
// lets the routing decision be unit-tested without a DSH runtime while still
// using the real message factory inside DSH.
let createUserMessageImpl = null;
async function resolveCreateUserMessage(injected) {
  if (typeof injected === "function") return injected;
  if (createUserMessageImpl) return createUserMessageImpl;
  const module = await import("@deepseek-ai/dsh-llm");
  createUserMessageImpl = module.createUserMessage;
  return createUserMessageImpl;
}

export const SIMILARITY_THRESHOLD = 0.6;
const GREETINGS = new Set([
  "你好",
  "您好",
  "早上好",
  "早安",
  "晚上好",
  "在吗",
  "嗨",
  "hi",
  "hello",
  "hey",
]);

function normalizedGreeting(text) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[，。！？!?.,\s]/gu, "");
}

/**
 * Legacy gate, used only when `router.enabled === false`.
 *
 * @param {string} query
 * @returns {boolean}
 */
export function shouldSkipPreRetrieval(query) {
  const text = String(query || "").trim();
  if (Array.from(text).length < 5) return true;
  return GREETINGS.has(normalizedGreeting(text));
}

export function topSimilarity(hits) {
  const top = Array.isArray(hits) ? hits[0] : null;
  if (!top) return 0;
  if (typeof top.similarity === "number" && Number.isFinite(top.similarity)) {
    return top.similarity;
  }
  // Chroma cosine distance: similarity = 1 - distance.
  if (typeof top.distance === "number" && Number.isFinite(top.distance)) {
    return 1 - top.distance;
  }
  return 0;
}

function memoryDate(hit) {
  const raw = Number(hit?.metadata?.timestamp_ms ?? hit?.timestamp_ms);
  if (!Number.isFinite(raw) || raw <= 0) return "unknown-date";
  return new Date(raw).toISOString().slice(0, 10);
}

export function createMemoryHint(result) {
  const hits = Array.isArray(result?.results)
    ? result.results.slice(0, 3)
    : [];
  const ordered = [...hits].sort((left, right) => {
    const leftTs = Number(left?.metadata?.timestamp_ms ?? left?.timestamp_ms) || 0;
    const rightTs =
      Number(right?.metadata?.timestamp_ms ?? right?.timestamp_ms) || 0;
    return leftTs - rightTs;
  });
  const lines = ordered
    .map((hit) => {
      const content = String(hit?.content || hit?.text || "").trim();
      if (!content) return "";
      const source = String(hit?.source_type || "memory");
      return `[${memoryDate(hit)}] ${source}: ${content}`;
    })
    .filter(Boolean);
  if (!lines.length) return "";
  return [
    "<memory-hint>",
    "[System Note: 以下是按时间顺序整理的历史记忆。若用户问及当前状态，请以最新记录为准；若问及流变过程，请说明前因后果。]",
    ...lines,
    "</memory-hint>",
  ].join("\n");
}

export function queryFromMessages(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || message.role !== "user") continue;
    const text = messageText(message);
    if (isUsableDialogueText("user", text)) return text.trim();
  }
  return "";
}

/** Format one router line without the raw query text. */
function routerLog(prefix, route, query) {
  const digest = hashQuery(query).slice(0, 8);
  const chars = Array.from(query).length;
  return (
    `${prefix} decision=${route.decision} source=${route.source} ` +
    `p=${Number(route.probabilityRetrieve ?? 0).toFixed(3)} ` +
    `latency=${route.latencyMs}ms budget=${route.budgetMs}ms ` +
    `degraded=${route.degraded === true} q_hash=${digest} chars=${chars}` +
    `${route.errorCode ? ` error=${route.errorCode}` : ""}` +
    `${route.fallbackReason ? ` fallback=${route.fallbackReason}` : ""}` +
    `${route.shadow ? " shadow=true" : ""}`
  );
}

export function registerPreRetrieval(
  ctx,
  {
    queryMemory,
    holoLog = () => {},
    diag = () => {},
    getConfig = () => ({}),
    router = null,
    createUserMessage = null,
  },
) {
  /** Injected or lazily loaded DSH message factory. */
  const messageFactory = () => resolveCreateUserMessage(createUserMessage);

  /**
   * Run the unchanged retrieval + injection path.
   *
   * @returns {Promise<object>} the (possibly extended) pre-step decision.
   */
  const retrieveAndInject = async (query, decision, signal) => {
    const result = await queryMemory(query, 3, 800, signal);
    if (result?.ok === false) {
      const line =
        `pre-retrieval unavailable: ${result.error || "unknown error"}; ` +
        `python=${result.python || "unknown"}; hf_home=${result.hf_home || "unknown"}`;
      diag(line);
      holoLog(line);
      return decision;
    }
    const hits = Array.isArray(result?.result?.results)
      ? result.result.results.slice(0, 3)
      : [];
    const score = topSimilarity(hits);
    if (score < SIMILARITY_THRESHOLD) {
      const line = `[PreRetrieval] skipped: top similarity ${score.toFixed(3)} < ${SIMILARITY_THRESHOLD}`;
      console.log(line);
      diag(line);
      return decision;
    }
    const hint = createMemoryHint(result?.result);
    if (!hint || signal?.aborted) {
      holoLog("Pre-retrieval skipped: no matching memories");
      return decision;
    }
    holoLog(`Pre-retrieval success: injected ${hits.length} memories`);
    // DSH's user/message event must carry role="user" (system-role messages
    // are rejected by the session schema). Keep the user role, but use a
    // non-user source kind: Chat projects non-"user" sources as `context`
    // nodes and excludes ordinary context from the transcript. This keeps the
    // hint model-visible without rendering it as a user bubble.
    const buildUserMessage = await messageFactory();
    const context = buildUserMessage({
      content: [{ type: "text", text: hint }],
      source: { kind: "memory-recall" },
    });
    return {
      ...decision,
      messages: [context, ...decision.messages],
    };
  };

  const onPreStep = async ({ messages, step, signal }, next) => {
    console.log(`[PreRetrieval] agent/pre-step fired step=${step}`);
    const decision = await next();
    if (decision.kind === "reject" || step !== 1) return decision;

    let config = {};
    try {
      config = getConfig() || {};
    } catch (error) {
      diag(`[PreRetrieval] config read failed: ${error?.stack || error}`);
    }
    if (config.enable_auto_recall === false) {
      console.log("[PreRetrieval] skipped: enable_auto_recall=false");
      return decision;
    }

    const query = queryFromMessages(messages);
    if (!query) {
      console.log("[PreRetrieval] no usable user query");
      return decision;
    }

    const routerConfig =
      config.router && typeof config.router === "object" ? config.router : {};
    const routerEnabled = router !== null && routerConfig.enabled !== false;

    // --- Revert path: exactly the pre-v0.4.0 behaviour -------------------
    if (!routerEnabled) {
      if (shouldSkipPreRetrieval(query)) {
        console.log(`[PreRetrieval] fast-pass: query=${query.slice(0, 80)}`);
        return decision;
      }
      console.log(`[PreRetrieval] query=${query.slice(0, 120)}`);
      return retrieveAndInject(query, decision, signal);
    }

    // --- Router arbitration ---------------------------------------------
    let route = null;
    try {
      route = await router.route({
        query,
        sessionId: decision.sessionId,
        turn: step,
        signal,
      });
    } catch (error) {
      // The engine promises never to throw; if it still does, "模糊即查".
      diag(`[Router] route threw, defaulting to retrieve: ${error?.stack || error}`);
      route = null;
    }

    if (!route) {
      diag("[Router] no route result; neutral default = retrieve");
      return retrieveAndInject(query, decision, signal);
    }
    if (route.cancelled === true || signal?.aborted) {
      console.log("[Router] cancelled; leaving the pre-step decision unchanged");
      return decision;
    }

    const line = routerLog("[Router]", route, query);
    if (route.shadow) {
      // Shadow: behave exactly like the old path, record what would happen.
      holoLog(`${line} (shadow: behavioural effect disabled)`);
      diag(line);
      return retrieveAndInject(query, decision, signal);
    }

    if (route.decision === "skip") {
      console.log(line);
      diag(line);
      return decision;
    }

    console.log(line);
    diag(line);
    return retrieveAndInject(query, decision, signal);
  };

  try {
    ctx.effect(
      () => ctx.on("agent/pre-step", onPreStep),
      "DSH-DREAM.pre-retrieval",
    );
    console.log("[PreRetrieval] agent/pre-step registered successfully");
    diag("[PreRetrieval] agent/pre-step registered successfully");
  } catch (error) {
    console.error("[PreRetrieval] agent/pre-step registration failed:", error);
    diag(
      `[PreRetrieval] agent/pre-step registration failed: ${error?.stack || error}`,
    );
  }
}
