/**
 * Config bridge normalization for DSH-DREAM.
 *
 * The Host settings form, the custom settings page, and the Python bridge file
 * speak three slightly different shapes:
 *
 * - the DSH `Config` form sends flat fields (`router_enabled`, …);
 * - the custom settings page posts the same flat fields plus preserved values;
 * - `agent_memory_config.json` stores a nested `router` block that the Python
 *   side and the RouterEngine consume.
 *
 * `normalizePayload` is the single mapping point. It is intentionally free of
 * DSH imports so the mapping can be unit-tested without a Host runtime.
 *
 * 中文说明：配置桥接的归一化层。把 DSH 表单 / 自绘设置页的扁平字段组装成后端
 * 使用的嵌套 `router` 配置，并在保存普通设置时保留未提交的路由配置。
 */

/** Read a Loader config field that may be a plain value or a volatile reference. */
export const current = (value) =>
  value !== null && typeof value === "object" && typeof value.get === "function"
    ? value.get()
    : value;

const CLOUD_FLAT_KEYS = [
  ["router_cloud_base_url", "base_url"],
  ["router_cloud_path", "path"],
  ["router_cloud_model", "model"],
  ["router_cloud_api_key_env", "api_key_env"],
];

/**
 * UI-only holographic indicator defaults.
 *
 * `enabled` is the interface master switch. It never participates in the
 * write decision; `holographic_mode` remains the only write switch.
 */
export const HOLO_INDICATOR_DEFAULT = Object.freeze({
  enabled: true,
});

/**
 * Normalize the nested `holo_indicator` block from the Config form, the
 * settings descriptor, or the bridge file. A missing block (old config) falls
 * back to the documented default; an old `style` key is ignored so removing
 * the theme variant never breaks an existing bridge file.
 *
 * @param {unknown} value Raw indicator value (possibly a volatile reference).
 * @param {unknown} [previous] Previously normalized indicator block.
 * @returns {{ enabled: boolean }}
 */
export function normalizeHoloIndicator(value, previous = null) {
  const currentValue = current(value);
  const previousValue = current(previous);
  const source =
    currentValue && typeof currentValue === "object" ? currentValue : {};
  const fallback =
    previousValue && typeof previousValue === "object" ? previousValue : {};
  const enabled = current(source.enabled ?? fallback.enabled);
  return {
    enabled:
      enabled === undefined || enabled === null
        ? HOLO_INDICATOR_DEFAULT.enabled
        : Boolean(enabled),
  };
}

/**
 * Assemble the nested `router` block from flat `router_*` fields.
 *
 * @param {object} value Raw form/config value.
 * @param {object|null} previous Previously normalized bridge payload.
 * @returns {object}
 */
export function normalizeRouterPayload(value, previous = null) {
  const previousRouter =
    previous?.router && typeof previous.router === "object" ? previous.router : {};
  const provided =
    value?.router && typeof value.router === "object" && current(value.router) !== null
      ? current(value.router)
      : null;
  const nested = provided || previousRouter;
  const nestedRegex =
    nested.regex && typeof nested.regex === "object" ? nested.regex : {};
  const nestedLocal =
    nested.local && typeof nested.local === "object" ? nested.local : {};
  const nestedCloud =
    nested.cloud && typeof nested.cloud === "object" ? nested.cloud : {};
  const nestedCache =
    nested.cache && typeof nested.cache === "object" ? nested.cache : {};
  const nestedCircuit =
    nested.circuit && typeof nested.circuit === "object" ? nested.circuit : {};
  const nestedAudit =
    nested.audit && typeof nested.audit === "object" ? nested.audit : {};

  const flatBoolean = (flatKey, fallback) => {
    const raw = current(value?.[flatKey]);
    if (raw === undefined || raw === null) return Boolean(fallback);
    if (typeof raw === "string") {
      const text = raw.trim().toLowerCase();
      if (["1", "true", "yes", "on"].includes(text)) return true;
      if (["0", "false", "no", "off"].includes(text)) return false;
    }
    return Boolean(raw);
  };
  const flatString = (flatKey, fallback) => {
    const raw = current(value?.[flatKey]);
    if (raw === undefined || raw === null) return fallback;
    const text = String(raw).trim();
    return text || fallback;
  };
  const flatNumber = (flatKey, fallback) => {
    const raw = Number(current(value?.[flatKey]));
    return Number.isFinite(raw) ? raw : fallback;
  };

  const cloudValues = {};
  let hasCloudFields = false;
  for (const [flatKey, target] of CLOUD_FLAT_KEYS) {
    const raw = current(value?.[flatKey]);
    const text =
      raw === undefined || raw === null ? "" : String(raw).trim();
    if (text) hasCloudFields = true;
    cloudValues[target] = text;
  }

  const explicitProvider = String(
    current(value?.router_cloud_provider) ?? "",
  ).trim();
  // A filled custom endpoint with no provider id means the generic "custom"
  // preset: the UI's advanced block is exactly that use case.
  const activeProvider =
    explicitProvider ||
    (hasCloudFields ? "custom" : String(nestedCloud.active ?? "").trim());

  const existingProviders =
    nestedCloud.providers && typeof nestedCloud.providers === "object"
      ? nestedCloud.providers
      : {};
  const providers = { ...existingProviders };
  if (activeProvider) {
    const base = { ...(providers[activeProvider] || {}) };
    for (const [target, text] of Object.entries(cloudValues)) {
      if (text) base[target] = text;
    }
    providers[activeProvider] = base;
  }

  return {
    enabled: flatBoolean("router_enabled", nested.enabled ?? true),
    mode: flatString("router_mode", String(nested.mode ?? "auto")),
    budget_ms: flatNumber("router_budget_ms", Number(nested.budget_ms ?? 200)),
    threshold: flatNumber("router_threshold", Number(nested.threshold ?? 0.55)),
    uncertainty_band: flatNumber(
      "router_uncertainty_band",
      Number(nested.uncertainty_band ?? 0.1),
    ),
    timeout_policy: flatString(
      "router_timeout_policy",
      String(nested.timeout_policy ?? "retrieve"),
    ),
    error_policy: String(nested.error_policy ?? "fallback_regex"),
    on_internal_error: String(nested.on_internal_error ?? "retrieve"),
    shadow_mode: flatBoolean("router_shadow_mode", nested.shadow_mode ?? false),
    regex: {
      enabled: nestedRegex.enabled !== false,
      default_decision:
        nestedRegex.default_decision === "skip" ? "skip" : "retrieve",
      explicit_fast_path: nestedRegex.explicit_fast_path === true,
    },
    local: {
      backend: flatString(
        "router_local_backend",
        String(nestedLocal.backend ?? "bge-head"),
      ),
      hedge: nestedLocal.hedge !== false,
      model_dir: String(nestedLocal.model_dir ?? "").trim(),
      warmup_timeout_ms: Number(nestedLocal.warmup_timeout_ms ?? 30000),
    },
    cloud: {
      enabled: flatBoolean("router_cloud_enabled", nestedCloud.enabled ?? false),
      active: activeProvider,
      budget_ms: Number(nestedCloud.budget_ms ?? 150),
      providers,
    },
    cache: {
      enabled: nestedCache.enabled !== false,
      ttl_ms: Number(nestedCache.ttl_ms ?? 120000),
      max_entries: Number(nestedCache.max_entries ?? 256),
    },
    circuit: {
      failure_threshold: Number(nestedCircuit.failure_threshold ?? 3),
      cooldown_ms: Number(nestedCircuit.cooldown_ms ?? 300000),
    },
    audit: {
      enabled: nestedAudit.enabled !== false,
      store_query: "hash",
      max_bytes: Number(nestedAudit.max_bytes ?? 4194304),
    },
  };
}

/**
 * Normalize a raw Config, settings descriptor, or HTTP payload for the bridge.
 *
 * @param {object} raw
 * @param {object|null} [previous] Existing normalized bridge payload, used to
 *   keep router settings the settings form did not send.
 * @returns {object}
 */
export function normalizePayload(raw, previous = null) {
  const value = raw && typeof raw === "object" ? raw : {};
  const provider = String(current(value.embedding_provider) ?? "local")
    .trim()
    .toLowerCase();
  return {
    enable_memory: Boolean(current(value.enable_memory) ?? true),
    enable_auto_recall: Boolean(current(value.enable_auto_recall) ?? true),
    // Preserve an explicit UI override, but never synthesize a default path:
    // the backend resolves the per-user DSH data directory (and honors
    // AGENT_MEMORY_RAG_PATH) when this field is blank.
    rag_path: String(current(value.rag_path) ?? "").trim(),
    python_path: String(current(value.python_path) ?? "").trim(),
    hf_home_path: String(current(value.hf_home_path) ?? "").trim(),
    max_window_tokens: Number(current(value.max_window_tokens) ?? 1500),
    retrieval_top_k: Number(current(value.retrieval_top_k) ?? 5),
    embedding_provider:
      provider === "openai_compatible" ? "openai_compatible" : "local",
    embedding_model: String(current(value.embedding_model) ?? "").trim(),
    embedding_api_key: String(current(value.embedding_api_key) ?? "").trim(),
    embedding_base_url: String(
      current(value.embedding_base_url) ?? "",
    ).trim(),
    auto_rebuild_collection: Boolean(
      current(value.auto_rebuild_collection) ?? true,
    ),
    holographic_mode: Boolean(current(value.holographic_mode) ?? false),
    // The indicator is a separate, UI-only switch: it must never be confused
    // with `holographic_mode`, and a legacy payload without the block keeps
    // the documented defaults.
    holo_indicator: normalizeHoloIndicator(
      value.holo_indicator,
      previous?.holo_indicator,
    ),
    router: normalizeRouterPayload(value, previous),
  };
}

/** Never write a bearer token into logs. */
export function redactPayload(payload) {
  return {
    ...payload,
    embedding_api_key: payload?.embedding_api_key ? "***" : "",
  };
}
