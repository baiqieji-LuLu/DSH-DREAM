/**
 * Cloud arbiter adapter for the v0.4.0 RouterEngine.
 *
 * Protocol: System One (`{"state": "...", "questions": {"need_rag": {...}}}`)
 * with an optional OpenAI-Decisions dialect.  Only the query text is sent, as
 * `state`; the question template is fixed so the query can never become an
 * instruction (see DESIGN_ROUTER.md 8.3).
 *
 * This module maps transport/HTTP failures onto the stable error-code table
 * from DESIGN_ROUTER.md 5.3 and never retries: decision APIs have no
 * idempotency key, so a retry can be billed twice.
 */

export const CLOUD_DISABLED = "CLOUD_DISABLED";
export const CLOUD_NOT_CONFIGURED = "CLOUD_NOT_CONFIGURED";
export const CLOUD_AUTH = "CLOUD_AUTH";
export const CLOUD_FORBIDDEN = "CLOUD_FORBIDDEN";
export const CLOUD_RATE_LIMIT = "CLOUD_RATE_LIMIT";
export const CLOUD_REGION = "CLOUD_REGION";
export const CLOUD_SERVER = "CLOUD_SERVER";
export const CLOUD_TIMEOUT = "CLOUD_TIMEOUT";
export const CLOUD_NETWORK = "CLOUD_NETWORK";
export const CLOUD_BAD_RESPONSE = "CLOUD_BAD_RESPONSE";
export const CLOUD_BUDGET_EXCEEDED = "CLOUD_BUDGET_EXCEEDED";

/** Provider presets; users fill in workspace-level endpoints themselves. */
export const PROVIDER_PRESETS = Object.freeze({
  typesafe: {
    id: "typesafe",
    displayName: "TypeSafe Jev",
    dialect: "systemone",
    baseUrl: "https://api.typesafe.ai",
    path: "/v1/systemone",
    model: "jev-latest",
    apiKeyEnv: "TYPESAFE_API_KEY",
    budgetMs: 150,
  },
  "ali-bailian": {
    id: "ali-bailian",
    displayName: "阿里百炼",
    dialect: "systemone",
    baseUrl: "",
    path: "/compatible-mode/v1/systemone",
    model: "decision-model-preview",
    apiKeyEnv: "DASHSCOPE_API_KEY",
    budgetMs: 150,
  },
  tokenrhythm: {
    id: "tokenrhythm",
    displayName: "TokenRhythm NeoHorse-Jev",
    dialect: "systemone",
    baseUrl: "https://tokenrhythm.studio/v1",
    path: "/systemone",
    model: "NeoHorse-Jev-4B",
    apiKeyEnv: "TOKENRHYTHM_API_KEY",
    budgetMs: 150,
  },
  openai: {
    id: "openai",
    displayName: "OpenAI Decisions (beta)",
    dialect: "openai-decisions",
    baseUrl: "https://api.openai.com",
    path: "/v1/decisions",
    model: "gpt-6-luna",
    apiKeyEnv: "OPENAI_API_KEY",
    budgetMs: 150,
  },
  "local-laya": {
    id: "local-laya",
    displayName: "Local Laya",
    dialect: "systemone",
    baseUrl: "http://localhost:8888",
    path: "/v1/systemone",
    model: "laya",
    apiKeyEnv: "",
    budgetMs: 150,
  },
  custom: {
    id: "custom",
    displayName: "Custom",
    dialect: "systemone",
    baseUrl: "",
    // Matches the settings-page placeholder: one endpoint URL plus an API-key
    // env var is enough to reach an OpenAI-compatible / System One endpoint.
    path: "/compatible-mode/v1/systemone",
    model: "decision-model-preview",
    apiKeyEnv: "",
    budgetMs: 150,
  },
});

export const NEED_RAG_INSTRUCTIONS =
  "判断这句话是否需要检索用户的长期记忆才能正确回答。需要回忆用户过去说过的事、偏好、经历或时间线时选 true；纯闲聊、通用知识、当前任务，或用户历史不影响回答时选 false。";

const MAX_STATE_CHARS = 2000;

/** Error carrying a stable router error code. */
export class CloudRouterError extends Error {
  /**
   * @param {string} code
   * @param {string} [message]
   * @param {{ status?: number, retryAfterMs?: number }} [extra]
   */
  constructor(code, message = "", extra = {}) {
    super(message || code);
    this.name = "CloudRouterError";
    this.code = code;
    this.status = extra.status;
    this.retryAfterMs = extra.retryAfterMs;
  }
}

/**
 * Map an HTTP status onto the documented error code.
 *
 * @param {number} status
 * @returns {string}
 */
export function classifyStatus(status) {
  if (status === 401) return CLOUD_AUTH;
  if (status === 403) return CLOUD_FORBIDDEN;
  if (status === 429) return CLOUD_RATE_LIMIT;
  if (status === 451) return CLOUD_REGION;
  if (status === 529 || status >= 500) return CLOUD_SERVER;
  if (status === 408) return CLOUD_TIMEOUT;
  return CLOUD_BAD_RESPONSE;
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return Math.min(1, Math.max(0, number));
}

/**
 * Normalize a provider answer into a calibrated probability.
 *
 * @param {unknown} payload
 * @returns {number|null}
 */
export function extractProbability(payload) {
  if (!payload || typeof payload !== "object") return null;
  const answers = payload.answers;
  if (answers && typeof answers === "object") {
    const answer = answers.need_rag ?? answers.needRag ?? answers[Object.keys(answers)[0]];
    if (answer && typeof answer === "object") {
      const direct = clamp01(
        answer.noul ?? answer.probability ?? answer.value ?? answer.score,
      );
      if (direct !== null) return direct;
      if (typeof answer.answer === "boolean") return answer.answer ? 1 : 0;
      const nested = answer.result;
      if (nested && typeof nested === "object") {
        const probability = clamp01(nested.noul ?? nested.probability ?? nested.value);
        if (probability !== null) return probability;
      }
    }
    if (typeof answer === "number") return clamp01(answer);
    if (typeof answer === "boolean") return answer ? 1 : 0;
  }
  const flat = clamp01(
    payload.probability ?? payload.noul ?? payload.need_rag_probability,
  );
  if (flat !== null) return flat;
  if (typeof payload.need_rag === "boolean") return payload.need_rag ? 1 : 0;
  if (payload.decision === "retrieve") return 1;
  if (payload.decision === "skip") return 0;
  return null;
}

/**
 * Build the fixed request body.  Only the query goes into `state`.
 *
 * @param {{ dialect: string, model: string, state: string, maxStateChars?: number }} args
 * @returns {object}
 */
export function buildRequestBody({ dialect, model, state, maxStateChars }) {
  const limit = Number.isFinite(maxStateChars)
    ? Math.max(1, Math.floor(maxStateChars))
    : MAX_STATE_CHARS;
  const truncated = String(state ?? "").slice(0, limit);
  if (dialect === "openai-decisions") {
    return {
      model,
      state: truncated,
      questions: {
        need_rag: {
          type: "yesno",
          instructions: NEED_RAG_INSTRUCTIONS,
          criteria: {
            true: "需要回忆用户过去说过的事、偏好、经历或时间线",
            false: "纯闲聊、通用知识、当前任务，或用户历史不影响回答",
          },
        },
      },
    };
  }
  return {
    model,
    state: truncated,
    questions: {
      need_rag: {
        type: "noul",
        instructions: NEED_RAG_INSTRUCTIONS,
        criteria: {
          true: "需要回忆用户过去说过的事、偏好、经历或时间线",
          false: "纯闲聊、通用知识、当前任务，或用户历史不影响回答",
        },
      },
    },
  };
}

/**
 * System One / OpenAI-Decisions cloud adjudicator.
 */
export class CloudRouter {
  /**
   * @param {object} options
   * @param {() => object} [options.getConfig] Returns the bridge payload.
   * @param {(envName: string) => (string|null|undefined|Promise<string|null|undefined>)} [options.resolveCredential]
   * @param {typeof fetch} [options.fetchImpl]
   * @param {(message: string) => void} [options.diag]
   */
  constructor({
    getConfig = () => ({}),
    resolveCredential = (name) => (name ? process.env?.[name] : null),
    fetchImpl = globalThis.fetch,
    diag = () => {},
  } = {}) {
    this.kind = "cloud";
    this.id = "cloud";
    this.getConfig = getConfig;
    this.resolveCredential = resolveCredential;
    this.fetchImpl = fetchImpl;
    this.diag = diag;
    this.lastLatencyMs = null;
    this.lastError = null;
  }

  /**
   * Resolve the active provider configuration from the bridge payload.
   *
   * @returns {null | {providerId: string, dialect: string, baseUrl: string,
   *   path: string, model: string, apiKeyEnv: string, budgetMs: number,
   *   maxStateChars: number, enabled: boolean}}
   */
  providerConfig() {
    let router = {};
    try {
      router = this.getConfig()?.router ?? {};
    } catch (error) {
      this.diag?.(`[Router] cloud config read failed: ${error?.message || error}`);
      return null;
    }
    const cloud = router.cloud && typeof router.cloud === "object" ? router.cloud : {};
    const activeId = String(cloud.active || "").trim();
    const providers =
      cloud.providers && typeof cloud.providers === "object" ? cloud.providers : {};
    const rawProvider =
      (activeId && providers[activeId]) || providers[Object.keys(providers)[0]] || {};
    const preset = PROVIDER_PRESETS[activeId] || PROVIDER_PRESETS.custom;
    const config = rawProvider && typeof rawProvider === "object" ? rawProvider : {};
    // An empty string means "not set" here, so a partially filled advanced form
    // still falls back to the preset path/model instead of posting to a bare
    // base URL.
    const pickText = (...values) => {
      for (const value of values) {
        const text = String(value ?? "").trim();
        if (text) return text;
      }
      return "";
    };
    const baseUrl = pickText(config.base_url, config.baseUrl, preset.baseUrl);
    const path = pickText(config.path, preset.path);
    const model = pickText(config.model, preset.model);
    const apiKeyEnv = pickText(
      config.api_key_env,
      config.apiKeyEnv,
      preset.apiKeyEnv,
    );
    const dialect = pickText(config.dialect, preset.dialect, "systemone");
    const budget = Number(config.budget_ms ?? cloud.budget_ms ?? preset.budgetMs ?? 150);
    return {
      providerId: activeId || preset.id,
      dialect: dialect === "openai-decisions" ? "openai-decisions" : "systemone",
      baseUrl,
      path,
      model,
      apiKeyEnv,
      budgetMs: Math.min(180, Math.max(40, Number.isFinite(budget) ? budget : 150)),
      maxStateChars: Number(config.max_state_chars) || MAX_STATE_CHARS,
      enabled: cloud.enabled === true,
    };
  }

  /**
   * Whether the cloud layer may participate right now (circuit is checked by
   * the engine; credentials are resolved lazily inside `route`).
   *
   * @returns {boolean}
   */
  isAvailable() {
    const config = this.providerConfig();
    if (!config) return false;
    if (!config.enabled) return false;
    if (!config.baseUrl || !config.model) return false;
    return true;
  }

  /**
   * Ask the provider whether this turn needs long-term memory.
   *
   * @param {{query: string}} input
   * @param {{deadlineMs: number, signal?: AbortSignal}} opts
   * @returns {Promise<{decision: "retrieve"|"skip", probabilityRetrieve: number,
   *   confidence: number, uncertain: boolean, reason: string,
   *   providerId: string, model: string}>}
   */
  async route(input, { deadlineMs, signal } = {}) {
    const config = this.providerConfig();
    if (!config) throw new CloudRouterError(CLOUD_DISABLED);
    if (!config.enabled) throw new CloudRouterError(CLOUD_DISABLED);
    if (!config.baseUrl || !config.model) {
      throw new CloudRouterError(CLOUD_NOT_CONFIGURED);
    }
    const apiKey = config.apiKeyEnv
      ? await this.resolveCredential(config.apiKeyEnv)
      : "";
    if (config.apiKeyEnv && !apiKey) {
      throw new CloudRouterError(CLOUD_NOT_CONFIGURED, "missing credential");
    }
    if (typeof this.fetchImpl !== "function") {
      throw new CloudRouterError(CLOUD_NETWORK, "fetch is unavailable");
    }
    const url = joinUrl(config.baseUrl, config.path);
    const body = JSON.stringify(
      buildRequestBody({
        dialect: config.dialect,
        model: config.model,
        state: input?.query ?? "",
        maxStateChars: config.maxStateChars,
      }),
    );
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(
      () => {
        timedOut = true;
        controller.abort();
      },
      Math.max(1, Number(deadlineMs) || config.budgetMs),
    );
    const onAbort = () => controller.abort();
    if (signal) {
      if (signal.aborted) {
        clearTimeout(timer);
        throw new CloudRouterError(CLOUD_TIMEOUT, "aborted before request");
      }
      signal.addEventListener?.("abort", onAbort, { once: true });
    }
    const started = nowMs();
    try {
      const response = await this.fetchImpl(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json",
          ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
        },
        body,
        signal: controller.signal,
      });
      const status = Number(response?.status) || 0;
      if (!response?.ok) {
        const retryAfterMs = parseRetryAfter(response?.headers);
        throw new CloudRouterError(classifyStatus(status), `HTTP ${status}`, {
          status,
          retryAfterMs,
        });
      }
      let payload;
      try {
        payload = await response.json();
      } catch {
        throw new CloudRouterError(CLOUD_BAD_RESPONSE, "response is not JSON");
      }
      const probability = extractProbability(payload);
      if (probability === null) {
        throw new CloudRouterError(CLOUD_BAD_RESPONSE, "missing need_rag answer");
      }
      this.lastLatencyMs = nowMs() - started;
      this.lastError = null;
      const threshold = Number(
        this.getConfig()?.router?.threshold ?? 0.55,
      );
      const band = Number(
        this.getConfig()?.router?.uncertainty_band ?? 0.1,
      );
      const low = Math.max(0, (Number.isFinite(threshold) ? threshold : 0.55) - (Number.isFinite(band) ? band : 0.1));
      const high = Math.min(1, (Number.isFinite(threshold) ? threshold : 0.55) + (Number.isFinite(band) ? band : 0.1));
      let decision = "skip";
      let uncertain = false;
      // Float tolerance: 0.55 - 0.10 must treat exactly 0.45 as in-band.
      if (probability >= high - 1e-9) decision = "retrieve";
      else if (probability < low - 1e-9) decision = "skip";
      else {
        // 模糊即查: the cloud abstains, the engine keeps arbitrating, and any
        // final neutral answer is still `retrieve`.
        decision = "retrieve";
        uncertain = true;
      }
      return {
        decision,
        probabilityRetrieve: probability,
        confidence: Math.max(probability, 1 - probability),
        uncertain,
        reason: uncertain ? "uncertain" : decision === "retrieve" ? "need_rag" : "no_need_rag",
        providerId: config.providerId,
        model: config.model,
      };
    } catch (error) {
      if (error instanceof CloudRouterError) {
        this.lastError = error.code;
        throw error;
      }
      this.lastError = timedOut ? CLOUD_TIMEOUT : CLOUD_NETWORK;
      if (timedOut || error?.name === "AbortError") {
        throw new CloudRouterError(CLOUD_TIMEOUT, "cloud slice deadline exceeded");
      }
      throw new CloudRouterError(CLOUD_NETWORK, error?.message || "network error");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener?.("abort", onAbort);
    }
  }
}

function joinUrl(baseUrl, path) {
  const base = String(baseUrl || "").replace(/\/+$/u, "");
  const suffix = String(path || "").startsWith("/") ? path : `/${path || ""}`;
  return `${base}${suffix}`;
}

function parseRetryAfter(headers) {
  try {
    const raw = headers?.get?.("retry-after");
    if (!raw) return undefined;
    const seconds = Number(raw);
    if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
    const date = Date.parse(raw);
    if (Number.isFinite(date)) return Math.max(0, date - Date.now());
  } catch {
    // Headers may be plain objects in tests.
  }
  return undefined;
}

function nowMs() {
  try {
    return globalThis.performance?.now?.() ?? Date.now();
  } catch {
    return Date.now();
  }
}

export default CloudRouter;
