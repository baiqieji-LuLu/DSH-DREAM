/**
 * Local router client for the v0.4.0 RouterEngine.
 *
 * This is a thin, non-blocking adapter over the Python bridge:
 *
 * - it never calls `bridge.ensure()` (that would provision `.dream-venv` or
 *   download a model from `agent/pre-step`);
 * - it reports `LOCAL_NOT_READY` / `LOCAL_BUSY` immediately instead of
 *   queueing behind `memory_search`, so the engine can degrade to regex;
 * - eligibility is optimistic: while warming, `op=route` itself is the probe,
 *   so a stale cached readiness flag can never pin the router to regex;
 * - `router.status` is only used to display state, not to gate routing.
 */

export const LOCAL_NOT_READY = "LOCAL_NOT_READY";
export const LOCAL_BUSY = "LOCAL_BUSY";
export const LOCAL_TIMEOUT = "LOCAL_TIMEOUT";
export const LOCAL_BUDGET_EXCEEDED = "LOCAL_BUDGET_EXCEEDED";
export const LOCAL_MODEL_MISMATCH = "LOCAL_MODEL_MISMATCH";
export const LOCAL_BACKEND_DISABLED = "LOCAL_BACKEND_DISABLED";
export const LOCAL_INTERNAL_ERROR = "INTERNAL";

/** Error carrying a stable local-router error code. */
export class LocalRouterError extends Error {
  /**
   * @param {string} code
   * @param {string} [message]
   */
  constructor(code, message = "") {
    super(message || code);
    this.name = "LocalRouterError";
    this.code = code;
  }
}

/**
 * Client for Python `op=route` / `op=router.status` / `op=router.warmup`.
 */
export class LocalRouterClient {
  /**
   * @param {object} options
   * @param {object} options.bridge Python bridge (see bridge_process.js).
   * @param {() => object} [options.getConfig]
   * @param {(message: string) => void} [options.diag]
   */
  constructor({ bridge, getConfig = () => ({}), diag = () => {} } = {}) {
    this.kind = "local";
    this.id = "local";
    this.bridge = bridge;
    this.getConfig = getConfig;
    this.diag = diag;
    this._ready = false;
    this._state = "unavailable";
    this._modelId = null;
    this._backend = null;
    this._dimension = 0;
    this._lastLatencyMs = null;
    this._lastError = null;
    this._warmupMs = null;
    // A model/dimension/SHA mismatch is the only condition that should stop
    // local attempts permanently.  "Warming" and transient busy/timeout are
    // recoverable, so they must never latch the backend off.
    this._fatal = false;
  }

  /** Configured backend name (`bge-head` / `onnx-intent` / `off`). */
  backendName() {
    try {
      const backend = this.getConfig()?.router?.local?.backend;
      return String(backend || "bge-head");
    } catch {
      return "bge-head";
    }
  }

  /**
   * Eligibility is deliberately optimistic: as long as the backend is not
   * switched off and did not fail fatally, the engine should try one
   * `op=route`.  The bridge answers `LOCAL_NOT_READY` in well under a
   * millisecond while Python is still warming, which is far more robust than
   * trusting a cached readiness flag (a stale flag used to pin the router to
   * the regex fallback forever).
   *
   * @returns {boolean}
   */
  isAvailable() {
    if (this.backendName() === "off") return false;
    if (this._fatal) return false;
    return true;
  }

  /** @returns {boolean} */
  get ready() {
    return this._ready === true;
  }

  /** @returns {string} */
  get state() {
    return this._state;
  }

  /** @returns {string|null} */
  get modelId() {
    return this._modelId;
  }

  /** @returns {Promise<object>} */
  async status(timeoutMs = 200) {
    if (!this.bridge || typeof this.bridge.routerStatus !== "function") {
      return this.describe();
    }
    try {
      const response = await this.bridge.routerStatus(timeoutMs);
      if (!response) {
        this._lastError = "LOCAL_TIMEOUT";
        return this.describe();
      }
      this._applyStatus(response);
      if (this._state !== "ready") {
        this.diag?.(
          `[Router] local status: state=${this._state} error=${this._lastError ?? "-"}`,
        );
      }
    } catch (error) {
      this._lastError = error?.code || error?.message || "LOCAL_STATUS_FAILED";
      this.diag?.(`[Router] local status failed: ${this._lastError}`);
    }
    return this.describe();
  }

  /**
   * Ask the bridge to warm the router.  Safe to call from plugin startup only;
   * never from `agent/pre-step`.
   *
   * @returns {Promise<object>}
   */
  async warmup() {
    if (!this.bridge || typeof this.bridge.warmupRouter !== "function") {
      this._lastError = LOCAL_NOT_READY;
      return { ok: false, code: LOCAL_NOT_READY };
    }
    try {
      const response = await this.bridge.warmupRouter();
      if (response && response.ok === false) {
        this._lastError = response.code || LOCAL_NOT_READY;
        return response;
      }
      this._state = String(response?.state || "warming");
      this._backend = response?.backend ?? this._backend;
      this._modelId = response?.model_id ?? this._modelId;
      if (this._state === "ready") this._ready = true;
      this._lastError = response?.error ?? null;
      return response || { ok: true, state: this._state };
    } catch (error) {
      this._lastError = error?.code || error?.message || LOCAL_NOT_READY;
      return { ok: false, code: this._lastError };
    }
  }

  /**
   * Classify one query locally.  Throws :class:`LocalRouterError` so the
   * engine can record `errorCode` and degrade without waiting.
   *
   * @param {{query: string}} input
   * @param {{deadlineMs: number, signal?: AbortSignal}} opts
   * @returns {Promise<{decision: "retrieve"|"skip", probabilityRetrieve: number,
   *   confidence: number, uncertain: boolean, reason: string,
   *   model: string|null, threshold: number, uncertaintyBand: number}>}
   */
  async route(input, { deadlineMs, signal } = {}) {
    if (this.backendName() === "off") {
      throw new LocalRouterError(LOCAL_BACKEND_DISABLED, "local backend is off");
    }
    if (this._fatal) {
      throw new LocalRouterError(LOCAL_MODEL_MISMATCH, this._lastError || "local router disabled");
    }
    if (!this.bridge || typeof this.bridge.routeQuery !== "function") {
      throw new LocalRouterError(LOCAL_NOT_READY, "bridge cannot route");
    }
    if (this.bridge.isBusy?.() === true) {
      throw new LocalRouterError(LOCAL_BUSY, "bridge is busy");
    }
    if (signal?.aborted) {
      throw new LocalRouterError(LOCAL_TIMEOUT, "aborted");
    }
    const response = await this.bridge.routeQuery(
      input?.query ?? "",
      deadlineMs,
      signal,
    );
    if (!response) {
      // No response is a transport/queue problem, not a model failure: keep
      // the backend eligible for the next turn.
      this._lastError = LOCAL_TIMEOUT;
      throw new LocalRouterError(LOCAL_TIMEOUT, "no response from bridge");
    }
    if (response.ok === false) {
      const code = String(response.code || LOCAL_NOT_READY);
      if (code === LOCAL_MODEL_MISMATCH) {
        this._fatal = true;
        this._ready = false;
        this._state = "failed";
      } else if (code === LOCAL_NOT_READY) {
        this._ready = false;
        this._state = "warming";
      }
      this._lastError = response.error || code;
      throw new LocalRouterError(code, response.error || code);
    }
    const probability = Number(response.probability);
    if (!Number.isFinite(probability)) {
      throw new LocalRouterError(LOCAL_INTERNAL_ERROR, "invalid probability");
    }
    this._ready = true;
    this._state = "ready";
    this._modelId = response.model_id ?? this._modelId;
    this._backend = response.backend ?? this._backend;
    this._lastLatencyMs = Number(response.latency_ms) || null;
    this._lastError = null;
    const decision = response.decision === "skip" ? "skip" : "retrieve";
    const threshold = Number(response.threshold);
    const band = Number(response.uncertainty_band);
    return {
      decision,
      probabilityRetrieve: Math.min(1, Math.max(0, probability)),
      confidence: Number(response.confidence) || Math.max(probability, 1 - probability),
      uncertain: response.uncertain === true,
      reason: response.uncertain ? "uncertain" : decision === "retrieve" ? "need_rag" : "no_need_rag",
      model: response.model_id ?? this._modelId,
      threshold: Number.isFinite(threshold) ? threshold : 0.55,
      uncertaintyBand: Number.isFinite(band) ? band : 0.1,
    };
  }

  /** @returns {object} */
  describe() {
    return {
      backend: this._backend || this.backendName(),
      ready: this._ready,
      state: this._state,
      modelId: this._modelId,
      dimension: this._dimension,
      warmupMs: this._warmupMs,
      lastLatencyMs: this._lastLatencyMs,
      lastError: this._lastError,
    };
  }

  _applyStatus(response) {
    if (!response || response.ok === false) {
      const code = response?.code || response?.error || LOCAL_NOT_READY;
      this._ready = false;
      this._fatal = code === LOCAL_MODEL_MISMATCH;
      this._state = this._fatal ? "failed" : "warming";
      this._lastError = code;
      return;
    }
    this._ready = response.ready === true;
    this._fatal = false;
    this._state = String(response.state || (this._ready ? "ready" : "unavailable"));
    this._backend = response.backend ?? this._backend;
    this._modelId = response.model_id ?? this._modelId;
    this._dimension = Number(response.dimension) || this._dimension;
    this._lastLatencyMs = Number(response.last_latency_ms) || this._lastLatencyMs;
    this._warmupMs = Number(response.warmup_ms) || this._warmupMs;
    this._lastError = response.error ?? null;
  }
}

export default LocalRouterClient;
