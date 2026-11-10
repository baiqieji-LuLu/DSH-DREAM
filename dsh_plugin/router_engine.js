/**
 * RouterEngine: the three-layer arbiter that decides whether one turn needs
 * long-term memory retrieval (DSH-DREAM v0.4.0, P0 + P1).
 *
 * Responsibilities, and nothing else:
 *
 *   guard (regex) -> cloud || local (hedged) -> local -> regex -> neutral rule
 *
 * - one hard 200 ms budget measured from `route()` entry;
 * - hedged backends: cloud and local start together, cloud is adopted first;
 * - neutral default: ambiguity, the deadline, and internal errors all resolve
 *   to `retrieve` ("模糊即查": better to over-retrieve than to forget);
 * - circuit breaker, TTL+LRU cache, and an audit log that stores only a query
 *   hash by default.
 *
 * The engine always resolves to a `RouteResult`; it never throws into
 * `agent/pre-step` and never emits dialogue text.
 */

import { appendFile, mkdir, rename, stat } from "node:fs/promises";
import { createHash, randomUUID } from "node:crypto";
import { dirname } from "node:path";
import {
  RegexRouter,
  RULES_VERSION,
  normalizeQuery,
} from "./router_regex.js";

export { normalizeQuery };

export const DECISION_RETRIEVE = "retrieve";
export const DECISION_SKIP = "skip";
export const SOURCE_CLOUD = "cloud";
export const SOURCE_LOCAL = "local";
export const SOURCE_REGEX = "regex";
export const SOURCE_FALLBACK = "fallback";

export const DEFAULT_BUDGET_MS = 200;
/** Timer jitter and Promise scheduling reserve (DESIGN_ROUTER.md 4.3). */
export const RESERVE_MS = 13;
export const MIN_CLOUD_SLICE_MS = 40;
export const MIN_LOCAL_SLICE_MS = 15;
export const MAX_BUDGET_MS = 200;

const ROUTER_MODES = new Set(["auto", "cloud", "local", "regex"]);
const POLICIES = new Set(["bypass_main", "retrieve", "skip"]);
const LOCAL_BACKENDS = new Set(["bge-head", "onnx-intent", "off"]);
const STORE_QUERY_MODES = new Set(["hash", "full", "none"]);

export const ROUTER_CONFIG_DEFAULTS = Object.freeze({
  enabled: true,
  mode: "auto",
  budgetMs: DEFAULT_BUDGET_MS,
  threshold: 0.55,
  uncertaintyBand: 0.1,
  timeoutPolicy: "retrieve",
  errorPolicy: "fallback_regex",
  onInternalError: "retrieve",
  shadowMode: false,
  regex: Object.freeze({
    enabled: true,
    defaultDecision: "retrieve",
    explicitFastPath: false,
  }),
  local: Object.freeze({
    backend: "bge-head",
    hedge: true,
    modelDir: "",
    warmupTimeoutMs: 30000,
  }),
  cloud: Object.freeze({ enabled: false, active: "", budgetMs: 150, providers: {} }),
  cache: Object.freeze({ enabled: true, ttlMs: 120000, maxEntries: 256 }),
  circuit: Object.freeze({ failureThreshold: 3, cooldownMs: 300000 }),
  audit: Object.freeze({ enabled: true, storeQuery: "hash", maxBytes: 4194304 }),
});

function bool(value, fallback) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(text)) return true;
    if (["0", "false", "no", "off"].includes(text)) return false;
  }
  if (value === 0 || value === 1) return Boolean(value);
  return fallback;
}

function number(value, fallback, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(minimum, Math.min(maximum, parsed));
}

function choice(value, fallback, allowed) {
  const text = String(value ?? "").trim().toLowerCase();
  return allowed.has(text) ? text : fallback;
}

function policy(value, fallback) {
  return choice(value, fallback, POLICIES);
}

/**
 * Normalize the nested `router` block (or flat `router_*` fields) into the
 * engine's canonical shape.  Unknown values fall back to safe defaults.
 *
 * @param {unknown} raw Bridge payload, nested router block, or flat config.
 * @returns {object}
 */
export function normalizeRouterConfig(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const nested =
    source.router && typeof source.router === "object" ? source.router : source;
  const regex = nested.regex && typeof nested.regex === "object" ? nested.regex : {};
  const local = nested.local && typeof nested.local === "object" ? nested.local : {};
  const cloud = nested.cloud && typeof nested.cloud === "object" ? nested.cloud : {};
  const cache = nested.cache && typeof nested.cache === "object" ? nested.cache : {};
  const circuit =
    nested.circuit && typeof nested.circuit === "object" ? nested.circuit : {};
  const audit = nested.audit && typeof nested.audit === "object" ? nested.audit : {};

  const pick = (flatKey, nestedValue, fallback) => {
    if (source[flatKey] !== undefined) return source[flatKey];
    if (nestedValue !== undefined && nestedValue !== null) return nestedValue;
    return fallback;
  };

  const config = {
    enabled: bool(pick("router_enabled", nested.enabled, true), true),
    mode: choice(pick("router_mode", nested.mode, "auto"), "auto", ROUTER_MODES),
    budgetMs: Math.round(
      number(
        pick("router_budget_ms", nested.budget_ms, DEFAULT_BUDGET_MS),
        DEFAULT_BUDGET_MS,
        50,
        MAX_BUDGET_MS,
      ),
    ),
    threshold: number(
      pick("router_threshold", nested.threshold, 0.55),
      0.55,
      0,
      1,
    ),
    uncertaintyBand: number(
      pick("router_uncertainty_band", nested.uncertainty_band, 0.1),
      0.1,
      0,
      0.3,
    ),
    timeoutPolicy: policy(
      pick("router_timeout_policy", nested.timeout_policy, "retrieve"),
      "retrieve",
    ),
    errorPolicy: policy(
      pick("router_error_policy", nested.error_policy, "fallback_regex"),
      "fallback_regex",
    ),
    onInternalError: policy(
      pick("router_on_internal_error", nested.on_internal_error, "retrieve"),
      "retrieve",
    ),
    shadowMode: bool(pick("router_shadow_mode", nested.shadow_mode, false), false),
    regex: {
      enabled: bool(regex.enabled, true),
      defaultDecision: choice(regex.default_decision, "retrieve", new Set(["retrieve", "skip"])),
      explicitFastPath:
        bool(regex.explicit_fast_path, false) ||
        bool(pick("router_regex_explicit_fast_path", undefined, false), false),
    },
    local: {
      backend: choice(local.backend, "bge-head", LOCAL_BACKENDS),
      hedge: bool(local.hedge, true),
      modelDir: String(local.model_dir ?? "").trim(),
      warmupTimeoutMs: Math.round(
        number(local.warmup_timeout_ms, 30000, 1000, 120000),
      ),
    },
    cloud: {
      enabled: bool(pick("router_cloud_enabled", cloud.enabled, false), false),
      active: String(cloud.active ?? pick("router_cloud_provider", undefined, "") ?? "").trim(),
      budgetMs: Math.round(number(cloud.budget_ms, 150, 40, 180)),
      providers:
        cloud.providers && typeof cloud.providers === "object" ? cloud.providers : {},
    },
    cache: {
      enabled: bool(cache.enabled, true),
      ttlMs: Math.round(number(cache.ttl_ms, 120000, 0, 600000)),
      maxEntries: Math.round(number(cache.max_entries, 256, 1, 4096)),
    },
    circuit: {
      failureThreshold: Math.round(
        number(circuit.failure_threshold, 3, 1, 10),
      ),
      cooldownMs: Math.round(number(circuit.cooldown_ms, 300000, 1000, 1800000)),
    },
    audit: {
      enabled: bool(audit.enabled, true),
      storeQuery: choice(audit.store_query, "hash", STORE_QUERY_MODES),
      maxBytes: Math.round(number(audit.max_bytes, 4194304, 65536, 67108864)),
    },
  };
  config.configHash = hashConfig(config);
  return config;
}

function hashConfig(config) {
  const salient = {
    mode: config.mode,
    budget: config.budgetMs,
    threshold: config.threshold,
    band: config.uncertaintyBand,
    timeout: config.timeoutPolicy,
    error: config.errorPolicy,
    internal: config.onInternalError,
    regex: config.regex,
    local: { backend: config.local.backend, modelDir: config.local.modelDir },
    cloud: { enabled: config.cloud.enabled, active: config.cloud.active, providers: config.cloud.providers },
    rules: RULES_VERSION,
  };
  return createHash("sha256").update(JSON.stringify(salient)).digest("hex");
}

/**
 * SHA-256 of the normalized query.  The audit log stores this unless
 * `audit.store_query="full"` is explicitly enabled.
 *
 * @param {unknown} text
 * @returns {string}
 */
export function hashQuery(text) {
  return createHash("sha256").update(normalizeQuery(text)).digest("hex");
}

function defaultNow() {
  try {
    return globalThis.performance?.now?.() ?? Date.now();
  } catch {
    return Date.now();
  }
}

function clamp01(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(1, Math.max(0, parsed));
}

/**
 * Small TTL + LRU cache.  Only confident model conclusions are stored.
 */
export class RouteCache {
  /**
   * @param {{ttlMs?: number, maxEntries?: number, now?: () => number}} [options]
   */
  constructor({ ttlMs = 120000, maxEntries = 256, now = defaultNow } = {}) {
    this.ttlMs = ttlMs;
    this.maxEntries = maxEntries;
    this._now = now;
    this._map = new Map();
  }

  /** @param {string} key */
  get(key) {
    const entry = this._map.get(key);
    if (!entry) return null;
    if (entry.expiresAt <= this._now()) {
      this._map.delete(key);
      return null;
    }
    this._map.delete(key);
    this._map.set(key, entry);
    return entry.value;
  }

  /**
   * @param {string} key
   * @param {object} value
   */
  set(key, value) {
    if (!(this.ttlMs > 0)) return;
    this._map.delete(key);
    this._map.set(key, { value, expiresAt: this._now() + this.ttlMs });
    while (this._map.size > this.maxEntries) {
      const oldest = this._map.keys().next().value;
      this._map.delete(oldest);
    }
  }

  clear() {
    this._map.clear();
  }

  /** @returns {number} */
  get size() {
    return this._map.size;
  }
}

/**
 * Circuit breaker with per-error cooldowns (DESIGN_ROUTER.md 4.4).
 */
export class CircuitBreaker {
  /**
   * @param {{failureThreshold?: number, cooldownMs?: number, now?: () => number}} [options]
   */
  constructor({ failureThreshold = 3, cooldownMs = 300000, now = defaultNow } = {}) {
    this.failureThreshold = failureThreshold;
    this.cooldownMs = cooldownMs;
    this._now = now;
    this._failures = 0;
    this._openedAt = null;
    this._openMs = 0;
    this._probeInFlight = false;
    this._lastCode = null;
    this._lastFailureAt = null;
  }

  /** @returns {"closed"|"open"|"half_open"} */
  get state() {
    if (this._openedAt === null) return "closed";
    if (this._now() - this._openedAt >= this._openMs) return "half_open";
    return "open";
  }

  /** @returns {boolean} */
  canAttempt() {
    const state = this.state;
    if (state === "closed") return true;
    if (state === "half_open") {
      if (this._probeInFlight) return false;
      this._probeInFlight = true;
      return true;
    }
    return false;
  }

  success() {
    this._failures = 0;
    this._openedAt = null;
    this._openMs = 0;
    this._probeInFlight = false;
    this._lastCode = null;
  }

  /**
   * @param {string} code
   * @param {number} [retryAfterMs]
   */
  failure(code, retryAfterMs) {
    const previous = this.state;
    this._failures += 1;
    this._lastCode = code || "CLOUD_NETWORK";
    this._lastFailureAt = this._now();
    this._probeInFlight = false;
    const cooldown = this.cooldownFor(code, retryAfterMs);
    if (previous === "half_open" || this._failures >= this.failureThreshold) {
      this._openedAt = this._now();
      this._openMs = cooldown;
    }
  }

  /**
   * @param {string} code
   * @param {number} [retryAfterMs]
   * @returns {number}
   */
  cooldownFor(code, retryAfterMs) {
    if (code === "CLOUD_AUTH" || code === "CLOUD_FORBIDDEN") return 15 * 60_000;
    if (code === "CLOUD_REGION") return 30 * 60_000;
    if (code === "CLOUD_RATE_LIMIT") {
      return Number.isFinite(retryAfterMs) && retryAfterMs > 0
        ? retryAfterMs
        : 60_000;
    }
    if (code === "CLOUD_SERVER" || code === "CLOUD_BAD_RESPONSE") return 2 * 60_000;
    if (code === "CLOUD_TIMEOUT" || code === "CLOUD_NETWORK") return 5 * 60_000;
    return this.cooldownMs;
  }

  /** @returns {object} */
  snapshot() {
    return {
      state: this.state,
      failures: this._failures,
      openedAt: this._openedAt,
      cooldownMs: this._openMs || this.cooldownMs,
      lastCode: this._lastCode,
      lastFailureAt: this._lastFailureAt,
    };
  }
}

/**
 * Append-only JSONL audit log.  Writes are queued and never block a decision.
 */
export class RouterAudit {
  /**
   * @param {{path?: string|null, storeQuery?: "hash"|"full"|"none",
   *   maxBytes?: number, write?: (line: string) => Promise<void>,
   *   diag?: (message: string) => void, nowWall?: () => Date}} [options]
   */
  constructor({
    path = null,
    storeQuery = "hash",
    maxBytes = 4194304,
    write = null,
    diag = () => {},
    nowWall = () => new Date(),
  } = {}) {
    this.path = path;
    this.storeQuery = STORE_QUERY_MODES.has(storeQuery) ? storeQuery : "hash";
    this.maxBytes = maxBytes;
    this.diag = diag;
    this.nowWall = nowWall;
    this._write = write;
    this._queue = Promise.resolve();
    this._bytes = null;
    this._ensureDir = null;
  }

  /**
   * Build one audit record.  Query text is replaced by a digest unless
   * `storeQuery === "full"`.
   *
   * @param {object} entry
   * @returns {object}
   */
  buildRecord(entry) {
    const query = String(entry.query ?? "");
    const record = {
      ts: this.nowWall().toISOString(),
      trace_id: entry.traceId,
      session_id: entry.sessionId ?? null,
      turn: entry.turn ?? null,
      query_hash: this.storeQuery === "none" ? null : hashQuery(query),
      query_chars: Array.from(query).length,
      decision: entry.decision,
      source: entry.source,
      provider: entry.providerId ?? null,
      model: entry.model ?? null,
      probability: entry.probabilityRetrieve,
      confidence: entry.confidence,
      threshold: entry.threshold,
      uncertainty_band: entry.uncertaintyBand,
      uncertain: entry.uncertain === true,
      reason: entry.reason ?? null,
      latency_ms: entry.latencyMs,
      budget_ms: entry.budgetMs,
      degraded: entry.degraded === true,
      cache_hit: entry.cacheHit === true,
      fallback_reason: entry.fallbackReason ?? null,
      error_code: entry.errorCode ?? null,
      local_ready: entry.localReady === true,
      cloud_circuit: entry.cloudCircuit ?? null,
      shadow: entry.shadow === true,
      rules_version: RULES_VERSION,
    };
    if (this.storeQuery === "full") record.query = query;
    return record;
  }

  /**
   * Queue one record.  Returns immediately; call {@link flush} in tests.
   *
   * @param {object} entry
   */
  record(entry) {
    if (!this._write && !this.path) return;
    let line;
    try {
      line = `${JSON.stringify(this.buildRecord(entry))}\n`;
    } catch (error) {
      this.diag?.(`[Router] audit build failed: ${error?.message || error}`);
      return;
    }
    this._queue = this._queue
      .then(() => this._append(line))
      .catch((error) => {
        this.diag?.(`[Router] audit write failed: ${error?.message || error}`);
      });
  }

  /** Wait for every queued write (tests / close). */
  async flush() {
    await this._queue;
  }

  async _append(line) {
    if (this._write) {
      await this._write(line);
      return;
    }
    if (!this.path) return;
    this._ensureDir ??= mkdir(dirname(this.path), { recursive: true });
    await this._ensureDir;
    const lineBytes = Buffer.byteLength(line, "utf8");
    if (this._bytes === null) {
      try {
        this._bytes = (await stat(this.path)).size;
      } catch {
        this._bytes = 0; // missing file is the normal first-write case
      }
    }
    if (this._bytes + lineBytes > this.maxBytes) {
      try {
        await rename(this.path, `${this.path}.1`);
      } catch {
        // A failed rotation must not lose the new record.
      }
      this._bytes = 0;
    }
    await appendFile(this.path, line, "utf8");
    this._bytes += lineBytes;
  }
}

/**
 * The arbiter itself.
 */
export class RouterEngine {
  /**
   * @param {object} [options]
   * @param {() => object} [options.getConfig]
   * @param {import("./router_regex.js").RegexRouter} [options.regex]
   * @param {object|null} [options.cloud]
   * @param {object|null} [options.local]
   * @param {(message: string) => void} [options.diag]
   * @param {() => number} [options.now]
   * @param {() => string} [options.randomId]
   * @param {RouterAudit|null} [options.audit]
   * @param {string|null} [options.auditPath]
   */
  constructor({
    getConfig = () => ({}),
    regex = null,
    cloud = null,
    local = null,
    diag = () => {},
    now = defaultNow,
    randomId = () => randomUUID(),
    audit = null,
    auditPath = null,
  } = {}) {
    this.getConfig = getConfig;
    this.regex = regex || new RegexRouter();
    this.cloud = cloud;
    this.local = local;
    this.diag = diag;
    this._now = now;
    this._randomId = randomId;
    this._audit =
      audit ||
      new RouterAudit({
        path: auditPath,
        storeQuery: "hash",
        diag,
      });
    this._configHash = null;
    this._cache = new RouteCache({ now });
    this._circuit = new CircuitBreaker({ now });
    this._regexByHash = new Map();
    this._stats = {
      turns: 0,
      retrieve: 0,
      skip: 0,
      timeouts: 0,
      fallbacks: 0,
      cacheHits: 0,
      errors: 0,
      uncertain: 0,
      shadow: 0,
    };
  }

  // ------------------------------------------------------------------ public

  /**
   * Decide whether this turn needs long-term memory.  Never throws.
   *
   * @param {{query?: string, sessionId?: string, turn?: number,
   *   signal?: AbortSignal}} [input]
   * @returns {Promise<object>} RouteResult
   */
  async route(input = {}) {
    const context = {
      started: this._now(),
      traceId: this._randomId(),
      cacheHit: false,
      degraded: false,
      fallbackReason: null,
      errorCode: null,
      query: String(input?.query ?? ""),
      sessionId: input?.sessionId,
      turn: input?.turn,
    };
    this._stats.turns += 1;
    try {
      const config = this._config();
      context.config = config;
      context.deadline = context.started + config.budgetMs;
      if (config.shadowMode) this._stats.shadow += 1;
      const query = normalizeQuery(context.query);
      context.query = query;
      if (!query) {
        return this._finish(
          { decision: "skip", source: SOURCE_REGEX, reason: "empty", probabilityRetrieve: 0, confidence: 1, uncertain: false },
          context,
        );
      }
      const guard = config.regex.enabled
        ? this._regexFor(config).guard(query)
        : { kind: "neutral" };
      if (guard.kind === "hard_skip") {
        return this._finish(
          { decision: "skip", source: SOURCE_REGEX, reason: guard.reason, probabilityRetrieve: 0, confidence: 1, uncertain: false },
          context,
        );
      }
      if (guard.kind === "explicit" && config.regex.explicitFastPath) {
        return this._finish(
          { decision: "retrieve", source: SOURCE_REGEX, reason: "explicit_cue", probabilityRetrieve: 1, confidence: 1, uncertain: false },
          context,
        );
      }
      if (input?.signal?.aborted) return this._cancelled(context);

      const cacheKey = this._cacheKey(query, config);
      if (config.cache.enabled) {
        const cached = this._cache.get(cacheKey);
        if (cached) {
          context.cacheHit = true;
          this._stats.cacheHits += 1;
          return this._finish({ ...cached }, context);
        }
      }

      const cloudEligible = this._cloudEligible(config);
      const localAvailable = this._localAvailable(config);
      const hedgeLocal = localAvailable && (config.local.hedge || !cloudEligible);
      let localPromise = hedgeLocal
        ? this._runLocal(input, config, context.deadline)
        : null;
      let uncertain = null;

      if (cloudEligible) {
        const slice = this._cloudSlice(config, context.deadline);
        if (slice >= MIN_CLOUD_SLICE_MS) {
          const cloudCall = this.cloud.route(input, {
            deadlineMs: slice,
            signal: input?.signal,
          });
          const raced = await this._race(cloudCall, slice);
          if (raced.ok && raced.value) {
            const result = raced.value;
            this._circuit.success();
            if (result.uncertain !== true) {
              this._store(cacheKey, config, {
                decision: result.decision,
                source: SOURCE_CLOUD,
                providerId: result.providerId,
                model: result.model,
                probabilityRetrieve: result.probabilityRetrieve,
                confidence: result.confidence,
                reason: result.reason,
                threshold: config.threshold,
                uncertaintyBand: config.uncertaintyBand,
                uncertain: false,
              });
              return this._finish(
                {
                  ...result,
                  source: SOURCE_CLOUD,
                  threshold: config.threshold,
                  uncertaintyBand: config.uncertaintyBand,
                },
                context,
              );
            }
            uncertain = { ...result, source: SOURCE_CLOUD };
          } else {
            const code =
              raced.code === "TIMEOUT"
                ? "CLOUD_TIMEOUT"
                : raced.error?.code || "CLOUD_NETWORK";
            this._circuit.failure(code, raced.error?.retryAfterMs);
            this._noteError(context, code, "cloud");
          }
        } else {
          context.fallbackReason = "CLOUD_BUDGET_EXCEEDED";
        }
      }

      if (localAvailable) {
        const remaining = context.deadline - this._now() - RESERVE_MS;
        if (remaining >= MIN_LOCAL_SLICE_MS) {
          const pending =
            localPromise ?? this._runLocal(input, config, context.deadline);
          const outcome = await this._awaitLocal(pending, remaining);
          if (outcome.ok && outcome.result) {
            const result = outcome.result;
            if (result.uncertain !== true) {
              this._store(cacheKey, config, {
                decision: result.decision,
                source: SOURCE_LOCAL,
                model: result.model,
                probabilityRetrieve: result.probabilityRetrieve,
                confidence: result.confidence,
                reason: result.reason,
                threshold: result.threshold,
                uncertaintyBand: result.uncertaintyBand,
                uncertain: false,
              });
              return this._finish(
                {
                  decision: result.decision,
                  source: SOURCE_LOCAL,
                  model: result.model,
                  probabilityRetrieve: result.probabilityRetrieve,
                  confidence: result.confidence,
                  uncertain: false,
                  reason: result.reason,
                  threshold: result.threshold,
                  uncertaintyBand: result.uncertaintyBand,
                },
                context,
              );
            }
            uncertain = { ...result, source: SOURCE_LOCAL };
          } else {
            this._noteError(context, outcome.code || "LOCAL_NOT_READY", "local");
          }
        } else {
          context.fallbackReason = "LOCAL_BUDGET_EXCEEDED";
        }
      }

      if (input?.signal?.aborted) return this._cancelled(context);
      if (this._now() >= context.deadline) {
        return this._failOpen(config, context, "BUDGET_EXCEEDED");
      }

      // 模糊即查: a model abstained -> retrieve, regardless of where the
      // threshold band placed it.  This is the neutral default the task
      // requires (a 0.55/0.45 answer must not lose memory).
      if (uncertain) {
        this._stats.uncertain += 1;
        return this._finish(
          {
            decision: "retrieve",
            source: uncertain.source,
            model: uncertain.model ?? null,
            providerId: uncertain.providerId ?? null,
            probabilityRetrieve: uncertain.probabilityRetrieve,
            confidence: uncertain.confidence,
            uncertain: true,
            reason: "uncertain_default_retrieve",
            threshold: uncertain.threshold ?? config.threshold,
            uncertaintyBand: uncertain.uncertaintyBand ?? config.uncertaintyBand,
            fallbackReason: context.fallbackReason,
          },
          context,
        );
      }

      if (context.errorCode && config.errorPolicy !== "fallback_regex") {
        return this._policyDecision(config, context, config.errorPolicy);
      }

      const rule = this._regexFor(config).route(query, {
        defaultDecision: config.regex.defaultDecision,
      });
      return this._finish(
        {
          ...rule,
          source: SOURCE_REGEX,
          threshold: config.threshold,
          uncertaintyBand: config.uncertaintyBand,
        },
        context,
      );
    } catch (error) {
      this.diag?.(`[Router] internal error: ${error?.stack || error}`);
      let fallbackConfig;
      try {
        fallbackConfig = normalizeRouterConfig(this.getConfig());
      } catch {
        fallbackConfig = normalizeRouterConfig({});
      }
      return this._failOpen(
        fallbackConfig,
        {
          ...context,
          config: fallbackConfig,
          errorCode: "INTERNAL",
          startTime: context.started,
        },
        "INTERNAL",
      );
    }
  }

  /**
   * @returns {Promise<object>} RouterStatus
   */
  async status() {
    let config;
    try {
      config = this._config();
    } catch {
      config = normalizeRouterConfig({});
    }
    const cloudInfo = this._cloudStatus(config);
    const localInfo = this._localStatus(config);
    const primary = !config.enabled
      ? null
      : this._cloudEligible(config, { probe: false })
        ? SOURCE_CLOUD
        : localInfo.ready
          ? SOURCE_LOCAL
          : SOURCE_REGEX;
    const turns = this._stats.turns || 0;
    const rate = (value) => (turns > 0 ? value / turns : 0);
    return {
      enabled: config.enabled,
      mode: config.mode,
      state: !config.enabled
        ? "disabled"
        : localInfo.ready || cloudInfo.configured
          ? "ready"
          : "degraded",
      primary,
      budgetMs: config.budgetMs,
      threshold: config.threshold,
      uncertaintyBand: config.uncertaintyBand,
      shadowMode: config.shadowMode,
      cloud: cloudInfo,
      local: localInfo,
      regex: {
        enabled: config.regex.enabled,
        rulesVersion: RULES_VERSION,
        defaultDecision: config.regex.defaultDecision,
        explicitFastPath: config.regex.explicitFastPath,
      },
      stats: {
        turns,
        retrieveRate: rate(this._stats.retrieve),
        timeoutRate: rate(this._stats.timeouts),
        fallbackRate: rate(this._stats.fallbacks),
        cacheHitRate: rate(this._stats.cacheHits),
        uncertainRate: rate(this._stats.uncertain),
        counts: { ...this._stats },
      },
    };
  }

  /** Warm the local backend; safe to call from plugin startup only. */
  async warmup() {
    try {
      if (this._config().enabled === false) {
        this.diag?.("[Router] disabled; skipping local warmup");
        return this.status();
      }
      const local = this.local;
      if (local && typeof local.warmup === "function") {
        const result = await local.warmup();
        this.diag?.(`[Router] local warmup: ${JSON.stringify(result ?? {})}`);
      }
    } catch (error) {
      this.diag?.(`[Router] local warmup failed: ${error?.message || error}`);
    }
    return this.status();
  }

  /**
   * Drop cached conclusions after a configuration/model change.
   *
   * @param {string} [reason]
   */
  invalidate(reason = "manual") {
    this._cache.clear();
    this._configHash = null;
    this._regexByHash.clear();
    this.diag?.(`[Router] cache invalidated: ${reason}`);
  }

  /** Stop writers used by the engine. */
  close() {
    void this._audit?.flush?.();
  }

  // --------------------------------------------------------------- internals

  _config() {
    let raw;
    try {
      raw = this.getConfig();
    } catch (error) {
      this.diag?.(`[Router] config read failed: ${error?.message || error}`);
      raw = {};
    }
    const config = normalizeRouterConfig(raw);
    if (this._configHash !== null && this._configHash !== config.configHash) {
      this._cache.clear();
      this.invalidate("config_changed");
    }
    this._configHash = config.configHash;
    this._cache.ttlMs = config.cache.ttlMs;
    this._cache.maxEntries = config.cache.maxEntries;
    this._circuit.failureThreshold = config.circuit.failureThreshold;
    this._circuit.cooldownMs = config.circuit.cooldownMs;
    const auditStore = config.audit.enabled ? config.audit.storeQuery : "none";
    this._audit.storeQuery = STORE_QUERY_MODES.has(auditStore) ? auditStore : "hash";
    this._audit.maxBytes = config.audit.maxBytes;
    return config;
  }

  _regexFor(config) {
    const key = JSON.stringify({
      enabled: config.regex.enabled,
      defaultDecision: config.regex.defaultDecision,
      explicitFastPath: config.regex.explicitFastPath,
    });
    let router = this._regexByHash.get(key);
    if (!router) {
      router = new RegexRouter({
        enabled: config.regex.enabled,
        defaultDecision: config.regex.defaultDecision,
        explicitFastPath: config.regex.explicitFastPath,
      });
      this._regexByHash.set(key, router);
    }
    return router;
  }

  _cloudEligible(config, { probe = true } = {}) {
    if (!this.cloud || config.mode === "local" || config.mode === "regex") {
      return false;
    }
    if (config.cloud.enabled !== true) return false;
    let available = true;
    try {
      available = this.cloud.isAvailable?.() !== false;
    } catch {
      available = false;
    }
    if (!available) return false;
    return probe ? this._circuit.canAttempt() : this._circuit.state !== "open";
  }

  _localAvailable(config) {
    if (!this.local || config.mode === "regex" || config.local.backend === "off") {
      return false;
    }
    try {
      return this.local.isAvailable?.() === true;
    } catch {
      return false;
    }
  }

  _cloudSlice(config, deadline) {
    const remaining = deadline - this._now();
    return Math.min(config.cloud.budgetMs, remaining - RESERVE_MS - 20);
  }

  _cacheKey(query, config) {
    const localModel =
      (typeof this.local?.modelId === "string" && this.local.modelId) || "";
    const cloudConfig = (() => {
      try {
        return this.cloud?.providerConfig?.() ?? {};
      } catch {
        return {};
      }
    })();
    const cloudId = `${cloudConfig.providerId || config.cloud.active || ""}:${
      cloudConfig.model || ""
    }`;
    return createHash("sha256")
      .update(query)
      .update("\x1f")
      .update(config.configHash)
      .update("\x1f")
      .update(localModel)
      .update("\x1f")
      .update(cloudId)
      .digest("hex");
  }

  _store(cacheKey, config, value) {
    if (!config.cache.enabled) return;
    if (value.uncertain === true) return;
    this._cache.set(cacheKey, value);
  }

  _runLocal(input, config, deadline) {
    const local = this.local;
    if (!local) return Promise.resolve({ ok: false, code: "LOCAL_NOT_READY" });
    const remaining = Math.max(
      MIN_LOCAL_SLICE_MS,
      Number(deadline || 0) - this._now(),
    );
    return Promise.resolve()
      .then(() =>
        local.route(input, {
          deadlineMs: remaining,
          signal: input?.signal,
        }),
      )
      .then((result) => ({ ok: true, result }))
      .catch((error) => ({
        ok: false,
        code: error?.code || "LOCAL_NOT_READY",
        message: error?.message || String(error),
      }));
  }

  async _awaitLocal(pending, timeoutMs) {
    const raced = await this._race(pending, timeoutMs);
    if (!raced.ok) {
      return {
        ok: false,
        code:
          raced.code === "TIMEOUT"
            ? "LOCAL_TIMEOUT"
            : raced.code || "LOCAL_INTERNAL",
      };
    }
    return raced.value || { ok: false, code: "LOCAL_NOT_READY" };
  }

  _race(target, timeoutMs) {
    let promise;
    try {
      promise = Promise.resolve(target);
    } catch (error) {
      return Promise.resolve({ ok: false, code: error?.code || "ERROR", error });
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve(value);
      };
      const timer = setTimeout(
        () => finish({ ok: false, code: "TIMEOUT" }),
        Math.max(0, Number(timeoutMs) || 0),
      );
      promise.then(
        (value) => finish({ ok: true, value }),
        (error) =>
          finish({ ok: false, code: error?.code || "ERROR", error }),
      );
    });
  }

  _noteError(context, code, layer) {
    context.errorCode = code;
    context.degraded = true;
    context.fallbackReason = context.fallbackReason || `${layer}_${code}`;
    this._stats.errors += 1;
    this.diag?.(`[Router] ${layer} failed: ${code}`);
  }

  _policyDecision(config, context, policyName) {
    const decision =
      policyName === "retrieve"
        ? "retrieve"
        : policyName === "skip"
          ? "skip"
          : "skip"; // bypass_main = let the main model answer without RAG
    return this._finish(
      {
        decision,
        source: SOURCE_FALLBACK,
        reason: `error_policy_${policyName}`,
        probabilityRetrieve: decision === "retrieve" ? 1 : 0,
        confidence: 1,
        uncertain: false,
      },
      context,
    );
  }

  _failOpen(config, context, errorCode) {
    if (errorCode === "BUDGET_EXCEEDED") this._stats.timeouts += 1;
    this._stats.errors += 1;
    context.errorCode = errorCode;
    context.degraded = true;
    context.fallbackReason = context.fallbackReason || errorCode;
    const policyName =
      errorCode === "BUDGET_EXCEEDED" ? config.timeoutPolicy : config.onInternalError;
    const decision =
      policyName === "skip" || policyName === "bypass_main" ? "skip" : "retrieve";
    return this._finish(
      {
        decision,
        source: SOURCE_FALLBACK,
        reason: decision === "retrieve" ? "timeout_default_retrieve" : "timeout_bypass_main",
        probabilityRetrieve: decision === "retrieve" ? 1 : 0,
        confidence: 1,
        uncertain: false,
      },
      context,
    );
  }

  _cancelled(context) {
    context.degraded = true;
    context.fallbackReason = "ABORTED";
    return this._finish(
      {
        decision: "retrieve",
        source: SOURCE_FALLBACK,
        reason: "aborted",
        probabilityRetrieve: 0.5,
        confidence: 0.5,
        uncertain: true,
        cancelled: true,
      },
      context,
    );
  }

  _finish(result, context) {
    const config = context.config || normalizeRouterConfig({});
    const decision = result.decision === "skip" ? "skip" : "retrieve";
    const latencyMs = Math.max(0, this._now() - context.started);
    const out = {
      decision,
      source: result.source || SOURCE_FALLBACK,
      providerId: result.providerId ?? null,
      model: result.model ?? null,
      probabilityRetrieve: clamp01(
        result.probabilityRetrieve ?? (decision === "retrieve" ? 1 : 0),
      ),
      confidence: clamp01(
        result.confidence ?? (result.probabilityRetrieve ?? 0.5),
      ),
      uncertain: result.uncertain === true,
      reason: result.reason ?? null,
      threshold: result.threshold ?? config.threshold,
      uncertaintyBand: result.uncertaintyBand ?? config.uncertaintyBand,
      latencyMs: Math.round(latencyMs * 100) / 100,
      budgetMs: config.budgetMs,
      degraded: context.degraded === true,
      cacheHit: context.cacheHit === true,
      traceId: context.traceId,
      errorCode: context.errorCode ?? null,
      fallbackReason: context.fallbackReason ?? result.fallbackReason ?? null,
      shadow: config.shadowMode === true,
      cancelled: result.cancelled === true,
    };
    if (decision === "retrieve") this._stats.retrieve += 1;
    else this._stats.skip += 1;
    if (out.degraded) this._stats.fallbacks += 1;
    if (config.audit?.enabled !== false) {
      this._audit?.record?.({
        ...out,
        query: context.query,
        sessionId: context.sessionId,
        turn: context.turn,
        localReady: this._localAvailable(config),
        cloudCircuit: this._circuit.state,
      });
    }
    this.diag?.(
      `[Router] decision=${out.decision} source=${out.source} p=${out.probabilityRetrieve.toFixed(3)} ` +
        `latency=${out.latencyMs}ms budget=${out.budgetMs}ms degraded=${out.degraded}` +
        `${out.errorCode ? ` error=${out.errorCode}` : ""}` +
        `${out.shadow ? " shadow=true" : ""}`,
    );
    return out;
  }

  _cloudStatus(config) {
    let provider = null;
    try {
      provider = this.cloud?.providerConfig?.() ?? null;
    } catch {
      provider = null;
    }
    const env = String(provider?.apiKeyEnv || "").trim();
    let credential = "unknown";
    if (!env) credential = "not_required";
    else {
      try {
        credential = process.env?.[env] ? "configured" : "missing";
      } catch {
        credential = "unknown";
      }
    }
    return {
      configured: Boolean(provider && provider.enabled && provider.baseUrl && provider.model),
      providerId: provider?.providerId ?? null,
      model: provider?.model ?? null,
      dialect: provider?.dialect ?? null,
      credential,
      circuit: this._circuit.state,
      failures: this._circuit.snapshot().failures,
      lastLatencyMs: this.cloud?.lastLatencyMs ?? null,
      lastError: this.cloud?.lastError ?? null,
      eligible: this._cloudEligible({ ...config, cloud: { ...config.cloud } }, { probe: false }),
    };
  }

  _localStatus(config) {
    let described = {};
    try {
      described = this.local?.describe?.() ?? {};
    } catch {
      described = {};
    }
    return {
      backend: described.backend || config.local.backend,
      ready: described.ready === true,
      state: described.state ?? "unavailable",
      modelId: described.modelId ?? null,
      dimension: described.dimension ?? 0,
      warmupMs: described.warmupMs ?? null,
      lastLatencyMs: described.lastLatencyMs ?? null,
      lastError: described.lastError ?? null,
    };
  }
}

/**
 * @param {ConstructorParameters<typeof RouterEngine>[0]} [options]
 * @returns {RouterEngine}
 */
export function createRouterEngine(options = {}) {
  return new RouterEngine(options);
}

export default RouterEngine;
