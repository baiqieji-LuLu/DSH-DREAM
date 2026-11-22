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
 * The router never influences what gets injected.  The eligibility gate is
 * still similarity-based, but RRF-hybrid hits may pass at 0.5 and keyword-only
 * top hits are accepted (see `recallDecision`); `createMemoryHint` now states
 * explicitly that confirmed facts should be used before reading files.  An
 * explicit `memory_search` tool call never goes through here.
 */
import { isUsableDialogueText, messageText } from "./holographic_events.js";
import { appendRecallTelemetry } from "./recall_telemetry.js";
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
/**
 * RRF-hybrid hits are confirmed by both the vector and the BM25 retriever, so
 * they may pass at a slightly lower cosine score.  A short factual query such
 * as "浇水闹钟取消了吗" sits at ~0.59 while being the obvious top hit; dropping
 * it starves the model and makes it search files instead.
 */
export const FUSED_SIMILARITY_THRESHOLD = 0.5;
/** Gate wait must exceed real retrieval latency (observed 3.2-3.4 s). */
export const DEFAULT_PRE_RETRIEVAL_TIMEOUT_MS = 8000;
export const MIN_PRE_RETRIEVAL_TIMEOUT_MS = 100;
export const MAX_PRE_RETRIEVAL_TIMEOUT_MS = 60000;

/**
 * Resolve the injection gate's wait from the bridge config.
 *
 * @param {object} config Normalized bridge config.
 * @returns {number} Timeout in milliseconds, clamped to a safe range.
 */
export function resolvePreRetrievalTimeout(config) {
  const raw = Number(
    config?.pre_retrieval_timeout_ms ?? DEFAULT_PRE_RETRIEVAL_TIMEOUT_MS,
  );
  if (!Number.isFinite(raw)) return DEFAULT_PRE_RETRIEVAL_TIMEOUT_MS;
  return Math.max(
    MIN_PRE_RETRIEVAL_TIMEOUT_MS,
    Math.min(MAX_PRE_RETRIEVAL_TIMEOUT_MS, Math.round(raw)),
  );
}

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

/** Cosine similarity of one hit, or 0 when the retriever did not report one. */
export function hitSimilarity(hit) {
  if (!hit || typeof hit !== "object") return 0;
  if (typeof hit.similarity === "number" && Number.isFinite(hit.similarity)) {
    return hit.similarity;
  }
  // Chroma cosine distance: similarity = 1 - distance.
  if (typeof hit.distance === "number" && Number.isFinite(hit.distance)) {
    return 1 - hit.distance;
  }
  return 0;
}

/**
 * Best cosine similarity across the returned hits.
 *
 * The previous implementation only looked at hit[0]; RRF can put a BM25-only
 * hit first (no similarity at all), which then looked like 0 and silently
 * dropped a perfectly good recall.
 *
 * @param {Array<object>} hits
 * @returns {number}
 */
export function topSimilarity(hits) {
  const list = Array.isArray(hits) ? hits : [];
  let best = 0;
  for (const hit of list) {
    const score = hitSimilarity(hit);
    if (score > best) best = score;
  }
  return best;
}

/**
 * Decide whether a retrieved hit list is strong enough to inject.
 *
 * Rules, in order:
 * 1. a hit's cosine similarity >= 0.6 (classic vector gate);
 * 2. the top hit is `hybrid` (vector + BM25 agree) and >= 0.5;
 * 3. the top hit is BM25-only but has a positive RRF score (lexical match
 *    with no vector rank available);
 * 4. any hit's cosine similarity >= 0.6 (the top fused hit may be lexical).
 *
 * @param {Array<object>} hits
 * @returns {{inject: boolean, reason: string, score: number,
 *   retrievalSource: string|null}}
 */
export function recallDecision(hits) {
  const list = Array.isArray(hits) ? hits : [];
  if (!list.length) {
    return { inject: false, reason: "no_hits", score: 0, retrievalSource: null };
  }
  const top = list[0];
  const score = topSimilarity(list);
  const topScore = hitSimilarity(top);
  const topSource = String(top?.retrieval_source || "");
  if (topScore >= SIMILARITY_THRESHOLD) {
    return {
      inject: true,
      reason: "vector",
      score: topScore,
      retrievalSource: topSource || null,
    };
  }
  if (topSource === "hybrid" && topScore >= FUSED_SIMILARITY_THRESHOLD) {
    return {
      inject: true,
      reason: "hybrid",
      score: topScore,
      retrievalSource: topSource,
    };
  }
  if (topSource === "bm25" && Number(top?.rrf_score) > 0) {
    return {
      inject: true,
      reason: "keyword",
      score: topScore,
      retrievalSource: topSource,
    };
  }
  if (score >= SIMILARITY_THRESHOLD) {
    return {
      inject: true,
      reason: "vector_any_hit",
      score,
      retrievalSource: topSource || null,
    };
  }
  return {
    inject: false,
    reason: "below_threshold",
    score: Math.max(score, topScore),
    retrievalSource: topSource || null,
  };
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
    "[System Note: 以下是用户长期记忆中已经确认过的事实（按时间排序）。回答时请优先直接使用这些事实；除非用户明确要求核实，否则不要为了确认这些事实去读取文件或运行脚本。若与当前对话冲突，以用户在最新一轮的明确说法为准；若问及变化过程，请说明前因后果。]",
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

function recallRouterRecord(
  route,
  fallbackDecision = "retrieve",
  source = "legacy",
) {
  if (route) {
    return {
      decision: route.decision,
      source: route.source || "unknown",
      p: Number(route.probabilityRetrieve ?? 0),
      latency_ms: Number(route.latencyMs ?? 0),
      ...(route.shadow === true ? { shadow: true } : {}),
      ...(route.degraded === true ? { degraded: true } : {}),
    };
  }
  return {
    decision: fallbackDecision,
    source,
    p: fallbackDecision === "retrieve" ? 1 : 0,
    latency_ms: 0,
  };
}

function recallTimingRecord(result) {
  const timing = result?.result?.timing;
  if (!timing || typeof timing !== "object") return null;
  const out = {};
  for (const key of [
    "vectorize",
    "vector_scan",
    "bm25",
    "fusion",
    "cache",
    "total",
  ]) {
    const value = Number(timing[key]);
    out[key] = Number.isFinite(value) ? value : 0;
  }
  return out;
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
    recallTelemetry = appendRecallTelemetry,
  },
) {
  /**
   * Append one structured recall record. Telemetry must never break the gate.
   */
  const logRecall = (base, gate, timing = null) => {
    try {
      const record = {
        ts: new Date().toISOString(),
        turn: base.turn,
        query_hash: base.queryHash ?? null,
        query_chars: base.queryChars ?? 0,
        router: base.router ?? null,
        gate,
        timing,
      };
      const pending = recallTelemetry(record);
      if (pending && typeof pending.catch === "function") {
        pending.catch((error) =>
          diag(`[RecallTelemetry] append failed: ${error?.stack || error}`),
        );
      }
    } catch (error) {
      diag(`[RecallTelemetry] append failed: ${error?.stack || error}`);
    }
  };
  /** Injected or lazily loaded DSH message factory. */
  const messageFactory = () => resolveCreateUserMessage(createUserMessage);
  /** Monotonic pre-step counter for recall telemetry. */
  let turnCounter = 0;

  /**
   * Run the unchanged retrieval + injection path.
   *
   * @returns {Promise<object>} the (possibly extended) pre-step decision.
   */
  const retrieveAndInject = async (
    query,
    decision,
    signal,
    timeoutMs,
    base,
  ) => {
    const startedAt = Date.now();
    const result = await queryMemory(query, 3, timeoutMs, signal);
    const elapsedMs = Math.max(0, Date.now() - startedAt);
    if (signal?.aborted) {
      const line = `[PreRetrieval] cancelled: elapsed=${elapsedMs}ms timeout=${timeoutMs}ms`;
      console.log(line);
      diag(line);
      logRecall(base, {
        outcome: "cancelled",
        reason: "signal_aborted",
        score: 0,
        hits: 0,
        retrieval_ms: elapsedMs,
      });
      return decision;
    }
    if (!result || result.ok === false) {
      // Timeout, bridge failure, and a dependency error are "unavailable",
      // never "no_hits". Keep this branch before the recall gate so a slow
      // retrieval can no longer be reported as an empty result.
      const code = result?.code || "QUERY_TIMEOUT_OR_UNAVAILABLE";
      const detail =
        result?.error ||
        (code === "QUERY_TIMEOUT"
          ? `query timeout after ${result?.timeout_ms ?? timeoutMs}ms`
          : "bridge unavailable");
      const envSuffix =
        result?.python || result?.hf_home
          ? ` python=${result.python || "unknown"} ` +
            `hf_home=${result.hf_home || "unknown"}`
          : "";
      const line =
        `[PreRetrieval] unavailable (不可用): ${detail}; ` +
        `code=${code}; timeout=${timeoutMs}ms elapsed=${elapsedMs}ms ` +
        `(not a no-hit result)${envSuffix}`;
      console.warn(line);
      diag(line);
      holoLog(line);
      logRecall(base, {
        outcome: "unavailable",
        reason: code,
        score: 0,
        hits: 0,
        retrieval_ms: elapsedMs,
      });
      return decision;
    }
    const hits = Array.isArray(result?.result?.results)
      ? result.result.results.slice(0, 3)
      : [];
    const recall = recallDecision(hits);
    if (!recall.inject) {
      const line =
        `[PreRetrieval] skipped: ${recall.reason} ` +
        `score=${recall.score.toFixed(3)} vector>=${SIMILARITY_THRESHOLD} ` +
        `hybrid>=${FUSED_SIMILARITY_THRESHOLD} retrieval_ms=${elapsedMs}`;
      console.log(line);
      diag(line);
      logRecall(
        base,
        {
          outcome: "skip",
          reason: recall.reason,
          score: Number(recall.score ?? 0),
          hits: hits.length,
          retrieval_ms: elapsedMs,
        },
        recallTimingRecord(result),
      );
      return decision;
    }
    const hint = createMemoryHint(result?.result);
    if (!hint || signal?.aborted) {
      const line = `[PreRetrieval] skipped: empty_hint retrieval_ms=${elapsedMs}`;
      console.log(line);
      diag(line);
      holoLog(line);
      logRecall(
        base,
        {
          outcome: "skip",
          reason: "empty_hint",
          score: Number(recall.score ?? 0),
          hits: hits.length,
          retrieval_ms: elapsedMs,
        },
        recallTimingRecord(result),
      );
      return decision;
    }
    diag(
      `[PreRetrieval] inject: ${recall.reason} score=${recall.score.toFixed(3)} ` +
        `source=${recall.retrievalSource || "unknown"} hits=${hits.length} ` +
        `retrieval_ms=${elapsedMs}`,
    );
    holoLog(
      `Pre-retrieval success: injected ${hits.length} memories in ${elapsedMs}ms`,
    );
    logRecall(
      base,
      {
        outcome: "inject",
        reason: recall.reason,
        score: Number(recall.score ?? 0),
        hits: hits.length,
        retrieval_ms: elapsedMs,
      },
      recallTimingRecord(result),
    );
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
    turnCounter += 1;
    const turn = turnCounter;

    let config = {};
    try {
      config = getConfig() || {};
    } catch (error) {
      diag(`[PreRetrieval] config read failed: ${error?.stack || error}`);
    }
    if (config.enable_auto_recall === false) {
      console.log("[PreRetrieval] skipped: enable_auto_recall=false");
      logRecall(
        {
          turn,
          queryHash: null,
          queryChars: 0,
          router: recallRouterRecord(null, "skip", "config"),
        },
        {
          outcome: "skip",
          reason: "auto_recall_disabled",
          score: 0,
          hits: 0,
          retrieval_ms: 0,
        },
      );
      return decision;
    }
    const preRetrievalTimeoutMs = resolvePreRetrievalTimeout(config);

    const query = queryFromMessages(messages);
    if (!query) {
      console.log("[PreRetrieval] no usable user query");
      logRecall(
        {
          turn,
          queryHash: null,
          queryChars: 0,
          router: recallRouterRecord(null, "skip", "query"),
        },
        {
          outcome: "skip",
          reason: "no_query",
          score: 0,
          hits: 0,
          retrieval_ms: 0,
        },
      );
      return decision;
    }
    const queryHash = hashQuery(query);
    const queryChars = Array.from(query).length;
    const recallBase = (routerRecord) => ({
      turn,
      queryHash,
      queryChars,
      router: routerRecord,
    });

    const routerConfig =
      config.router && typeof config.router === "object" ? config.router : {};
    const routerEnabled = router !== null && routerConfig.enabled !== false;

    // --- Revert path: exactly the pre-v0.4.0 behaviour -------------------
    if (!routerEnabled) {
      if (shouldSkipPreRetrieval(query)) {
        console.log(`[PreRetrieval] fast-pass: query=${query.slice(0, 80)}`);
        logRecall(
          recallBase(recallRouterRecord(null, "skip", "legacy")),
          {
            outcome: "skip",
            reason: "short_or_greeting",
            score: 0,
            hits: 0,
            retrieval_ms: 0,
          },
        );
        return decision;
      }
      console.log(`[PreRetrieval] query=${query.slice(0, 120)}`);
      return retrieveAndInject(
        query,
        decision,
        signal,
        preRetrievalTimeoutMs,
        recallBase(recallRouterRecord(null, "retrieve", "legacy")),
      );
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
      return retrieveAndInject(
        query,
        decision,
        signal,
        preRetrievalTimeoutMs,
        recallBase(recallRouterRecord(null, "retrieve", "neutral")),
      );
    }
    if (route.cancelled === true || signal?.aborted) {
      console.log("[Router] cancelled; leaving the pre-step decision unchanged");
      logRecall(
        recallBase(recallRouterRecord(route)),
        {
          outcome: "cancelled",
          reason: "router_cancelled",
          score: 0,
          hits: 0,
          retrieval_ms: 0,
        },
      );
      return decision;
    }

    const line = routerLog("[Router]", route, query);
    if (route.shadow) {
      // Shadow: behave exactly like the old path, record what would happen.
      holoLog(`${line} (shadow: behavioural effect disabled)`);
      diag(line);
      return retrieveAndInject(
        query,
        decision,
        signal,
        preRetrievalTimeoutMs,
        recallBase(recallRouterRecord(route)),
      );
    }

    if (route.decision === "skip") {
      console.log(line);
      diag(line);
      logRecall(
        recallBase(recallRouterRecord(route)),
        {
          outcome: "skip",
          reason: route.fallbackReason || "router_skip",
          score: 0,
          hits: 0,
          retrieval_ms: 0,
        },
      );
      return decision;
    }

    console.log(line);
    diag(line);
    return retrieveAndInject(
      query,
      decision,
      signal,
      preRetrievalTimeoutMs,
      recallBase(recallRouterRecord(route)),
    );
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
