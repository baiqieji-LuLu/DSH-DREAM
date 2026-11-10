/**
 * RouterEngine unit + integration tests (v0.4.0, P0 + P1).
 *
 * Run with:  node tests/router_engine.test.mjs
 *
 * Focus: the neutral default (ambiguous / timeout / error -> retrieve), the
 * 200 ms budget, hedge/priority, circuit breaker, cache, audit hashing, and the
 * `router.enabled` / `shadow_mode` escape hatches in pre_retrieval.
 */

import assert from "node:assert/strict";
import {
  CircuitBreaker,
  RouteCache,
  RouterAudit,
  createRouterEngine,
  hashQuery,
  normalizeRouterConfig,
} from "../dsh_plugin/router_engine.js";
import {
  EXPLICIT,
  HARD_SKIP,
  NEUTRAL,
  RULES_VERSION,
  guard,
  normalizeQuery,
} from "../dsh_plugin/router_regex.js";
import { CloudRouterError } from "../dsh_plugin/router_cloud.js";
import { registerPreRetrieval } from "../dsh_plugin/pre_retrieval.js";

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const never = () => new Promise(() => {});

function deepMerge(base, patch) {
  if (!patch || typeof patch !== "object") return base;
  const out = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      base[key] &&
      typeof base[key] === "object"
    ) {
      out[key] = deepMerge(base[key], value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function routerConfig(patch = {}) {
  return deepMerge(
    {
      enabled: true,
      mode: "auto",
      budget_ms: 200,
      threshold: 0.55,
      uncertainty_band: 0.1,
      timeout_policy: "retrieve",
      error_policy: "fallback_regex",
      on_internal_error: "retrieve",
      shadow_mode: false,
      regex: { enabled: true, default_decision: "retrieve", explicit_fast_path: false },
      local: { backend: "bge-head", hedge: true, model_dir: "", warmup_timeout_ms: 30000 },
      cloud: { enabled: false, active: "", budget_ms: 150, providers: {} },
      cache: { enabled: true, ttl_ms: 120000, max_entries: 256 },
      circuit: { failure_threshold: 3, cooldown_ms: 300000 },
      audit: { enabled: false, store_query: "hash", max_bytes: 4194304 },
    },
    patch,
  );
}

function localRoute(overrides = {}) {
  return {
    decision: "retrieve",
    probabilityRetrieve: 0.9,
    confidence: 0.9,
    uncertain: false,
    reason: "need_rag",
    model: "fake:head:v1",
    threshold: 0.55,
    uncertaintyBand: 0.1,
    ...overrides,
  };
}

function fakeLocal({ available = true, impl = null } = {}) {
  const calls = [];
  return {
    calls,
    kind: "local",
    id: "local",
    modelId: "fake:head:v1",
    isAvailable: () => available,
    describe: () => ({
      backend: "bge-head",
      ready: available,
      state: available ? "ready" : "failed",
      modelId: "fake:head:v1",
      dimension: 8,
    }),
    warmup: async () => ({ ok: true, state: "ready" }),
    route: async (input, opts) => {
      calls.push({ input, opts });
      if (impl) return impl(input, opts);
      return localRoute();
    },
  };
}

function fakeCloud({ available = true, impl = null, prefix = "" } = {}) {
  const calls = [];
  const state = { lastLatencyMs: null, lastError: null };
  return {
    calls,
    state,
    kind: "cloud",
    id: "cloud",
    providerConfig: () => ({
      providerId: "fake-provider",
      model: "fake-decision",
      enabled: true,
      baseUrl: "https://example.invalid",
      path: "/v1/systemone",
      apiKeyEnv: "",
      dialect: "systemone",
      budgetMs: 150,
    }),
    isAvailable: () => available,
    route: async (input, opts) => {
      calls.push({ input, opts });
      if (impl) return impl(input, opts);
      return {
        decision: "retrieve",
        probabilityRetrieve: 0.92,
        confidence: 0.92,
        uncertain: false,
        reason: "need_rag",
        providerId: "fake-provider",
        model: "fake-decision",
        ...(prefix ? {} : {}),
      };
    },
  };
}

function makeEngine({ config, local = null, cloud = null, auditLines = null } = {}) {
  const audit = auditLines
    ? new RouterAudit({
        write: async (line) => auditLines.push(line),
        storeQuery: "hash",
      })
    : undefined;
  return createRouterEngine({
    getConfig: () => ({ router: config }),
    local,
    cloud,
    ...(audit ? { audit } : { auditPath: null }),
    diag: () => {},
  });
}

const tests = [];
function test(name, fn) {
  tests.push([name, fn]);
}

// ---------------------------------------------------------------- regex guard

test("regex guard hard-skips greetings, commands, suppression, and empty input", () => {
  assert.equal(guard("").kind, HARD_SKIP);
  assert.equal(guard("   ").kind, HARD_SKIP);
  assert.equal(guard("？？！").kind, HARD_SKIP);
  assert.equal(guard("你好！").kind, HARD_SKIP);
  assert.equal(guard("在吗").kind, HARD_SKIP);
  assert.equal(guard("/clear").kind, HARD_SKIP);
  assert.equal(guard("别翻旧账，我们聊点新的").kind, HARD_SKIP);
  assert.equal(guard("忽略记忆继续回答").kind, HARD_SKIP);
  assert.equal(guard("上次我们说的那个方案").kind, EXPLICIT);
  assert.equal(guard("帮我写一个快速排序函数").kind, NEUTRAL);
  assert.equal(normalizeQuery("  你  好  "), "你 好");
});

test("engine skips a greeting without touching any backend", async () => {
  const local = fakeLocal();
  const cloud = fakeCloud();
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    local,
    cloud,
  });
  const result = await engine.route({ query: "你好" });
  assert.equal(result.decision, "skip");
  assert.equal(result.source, "regex");
  assert.equal(result.reason, "greeting");
  assert.equal(local.calls.length, 0);
  assert.equal(cloud.calls.length, 0);
  assert.equal(result.shadow, false);
  assert.equal(result.latencyMs < 50, true);
});

test("engine skips empty input and honours explicit_fast_path", async () => {
  const engine = makeEngine({ config: routerConfig() });
  const empty = await engine.route({ query: "   " });
  assert.equal(empty.decision, "skip");

  const local = fakeLocal();
  const fast = makeEngine({
    config: routerConfig({ regex: { explicit_fast_path: true } }),
    local,
  });
  const result = await fast.route({ query: "上次那个蓝色马克杯在哪" });
  assert.equal(result.decision, "retrieve");
  assert.equal(result.source, "regex");
  assert.equal(local.calls.length, 0);
});

// --------------------------------------------------------------- arbitration

test("cloud confident beats a confident local (priority, hedge ignored)", async () => {
  const local = fakeLocal();
  const cloud = fakeCloud({
    impl: () => ({
      decision: "skip",
      probabilityRetrieve: 0.02,
      confidence: 0.98,
      uncertain: false,
      reason: "no_need_rag",
      providerId: "fake-provider",
      model: "fake-decision",
    }),
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    local,
    cloud,
  });
  const result = await engine.route({ query: "帮我写一个快速排序函数" });
  assert.equal(result.decision, "skip");
  assert.equal(result.source, "cloud");
  assert.equal(result.probabilityRetrieve, 0.02);
});

test("cloud abstains -> local arbitrates", async () => {
  const local = fakeLocal({ impl: () => localRoute({ probabilityRetrieve: 0.8 }) });
  const cloud = fakeCloud({
    impl: () => ({
      decision: "retrieve",
      probabilityRetrieve: 0.55,
      confidence: 0.55,
      uncertain: true,
      reason: "uncertain",
      providerId: "fake-provider",
      model: "fake-decision",
    }),
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    local,
    cloud,
  });
  const result = await engine.route({ query: "上次那个蓝色马克杯" });
  assert.equal(result.source, "local");
  assert.equal(result.decision, "retrieve");
  assert.equal(result.uncertain, false);
});

test("both layers abstain -> retrieve (中性犹豫即查)", async () => {
  const local = fakeLocal({
    impl: () => localRoute({ decision: "retrieve", probabilityRetrieve: 0.45, uncertain: true }),
  });
  const cloud = fakeCloud({
    impl: () => ({
      decision: "retrieve",
      probabilityRetrieve: 0.55,
      confidence: 0.55,
      uncertain: true,
      reason: "uncertain",
      providerId: "fake-provider",
      model: "fake-decision",
    }),
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    local,
    cloud,
  });
  const result = await engine.route({ query: "那个东西怎么样了" });
  assert.equal(result.decision, "retrieve");
  assert.equal(result.uncertain, true);
  assert.equal(result.reason, "uncertain_default_retrieve");
  assert.equal(result.source, "local");
});

test("cloud timeout -> retrieve within the 200 ms budget", async () => {
  const cloud = fakeCloud({ impl: () => never() });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud,
  });
  const started = Date.now();
  const result = await engine.route({ query: "帮我总结一下这篇文章" });
  const elapsed = Date.now() - started;
  assert.equal(result.decision, "retrieve");
  assert.equal(result.errorCode, "CLOUD_TIMEOUT");
  assert.equal(result.degraded, true);
  assert.equal(elapsed >= 120, true, `elapsed=${elapsed}`);
  assert.equal(elapsed < 400, true, `elapsed=${elapsed}`);
  assert.equal(result.source, "regex");
});

test("all layers time out -> still retrieve, never blocks past the budget", async () => {
  const local = fakeLocal({ impl: () => never() });
  const cloud = fakeCloud({ impl: () => never() });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true }, local: { hedge: false } }),
    local,
    cloud,
  });
  const started = Date.now();
  const result = await engine.route({ query: "帮我解释一下这个报错" });
  const elapsed = Date.now() - started;
  assert.equal(result.decision, "retrieve");
  assert.equal(result.degraded, true);
  assert.equal(elapsed < 400, true, `elapsed=${elapsed}`);
});

test("cloud auth error degrades without a retry", async () => {
  const cloud = fakeCloud({
    impl: () => {
      throw new CloudRouterError("CLOUD_AUTH", "HTTP 401", { status: 401 });
    },
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud,
  });
  const result = await engine.route({ query: "帮我看看这个配置" });
  assert.equal(result.decision, "retrieve");
  assert.equal(result.errorCode, "CLOUD_AUTH");
  assert.equal(cloud.calls.length, 1);
});

test("internal engine errors fail neutral instead of throwing", async () => {
  const local = fakeLocal({
    impl: () => {
      throw new Error("boom");
    },
  });
  const engine = createRouterEngine({
    getConfig: () => {
      throw new Error("config exploded");
    },
    local,
    diag: () => {},
  });
  const result = await engine.route({ query: "上次那个方案" });
  assert.equal(result.decision, "retrieve");
  assert.equal(typeof result.traceId, "string");
});

test("an already-aborted signal yields a cancelled result", async () => {
  const local = fakeLocal();
  const engine = makeEngine({ config: routerConfig(), local });
  const controller = new AbortController();
  controller.abort();
  const result = await engine.route({ query: "上次那个方案", signal: controller.signal });
  assert.equal(result.cancelled, true);
  assert.equal(local.calls.length, 0);
});

// ------------------------------------------------------------ circuit / cache

test("circuit opens after three failures and stops calling the cloud", async () => {
  const cloud = fakeCloud({
    impl: () => {
      throw new CloudRouterError("CLOUD_NETWORK", "dns");
    },
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud,
  });
  for (let i = 0; i < 4; i += 1) {
    const result = await engine.route({ query: `帮我处理第 ${i + 1} 条数据` });
    assert.equal(result.decision, "retrieve");
  }
  assert.equal(cloud.calls.length, 3);
  const status = await engine.status();
  assert.equal(status.cloud.circuit, "open");
});

test("CircuitBreaker uses per-error cooldowns", () => {
  let now = 0;
  const breaker = new CircuitBreaker({
    failureThreshold: 1,
    cooldownMs: 1000,
    now: () => now,
  });
  breaker.failure("CLOUD_AUTH");
  assert.equal(breaker.state, "open");
  const snapshot = breaker.snapshot();
  assert.equal(snapshot.cooldownMs, 15 * 60_000);
  now = 15 * 60_000;
  assert.equal(breaker.state, "half_open");
  assert.equal(breaker.canAttempt(), true);
  assert.equal(breaker.canAttempt(), false); // only one probe
  breaker.failure("CLOUD_AUTH");
  assert.equal(breaker.state, "open");
});

test("RouteCache expires entries and evicts the oldest", () => {
  let now = 0;
  const cache = new RouteCache({ ttlMs: 100, maxEntries: 2, now: () => now });
  cache.set("a", { decision: "retrieve" });
  cache.set("b", { decision: "skip" });
  cache.set("c", { decision: "retrieve" });
  assert.equal(cache.size, 2);
  assert.equal(cache.get("a"), null); // evicted
  assert.equal(cache.get("c").decision, "retrieve");
  now = 101;
  assert.equal(cache.get("c"), null); // expired
});

test("confident cloud conclusions are cached; abstentions are not", async () => {
  const cloud = fakeCloud({
    impl: () => ({
      decision: "skip",
      probabilityRetrieve: 0.05,
      confidence: 0.95,
      uncertain: false,
      reason: "no_need_rag",
      providerId: "fake-provider",
      model: "fake-decision",
    }),
  });
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud,
  });
  const first = await engine.route({ query: "帮我写一个快速排序函数" });
  const second = await engine.route({ query: "帮我写一个快速排序函数" });
  assert.equal(first.cacheHit, false);
  assert.equal(second.cacheHit, true);
  assert.equal(second.decision, "skip");
  assert.equal(cloud.calls.length, 1);

  const uncertainCloud = fakeCloud({
    impl: () => ({
      decision: "retrieve",
      probabilityRetrieve: 0.52,
      confidence: 0.52,
      uncertain: true,
      reason: "uncertain",
      providerId: "fake-provider",
      model: "fake-decision",
    }),
  });
  const local = fakeLocal({
    impl: () => localRoute({ probabilityRetrieve: 0.48, uncertain: true }),
  });
  const engine2 = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud: uncertainCloud,
    local,
  });
  await engine2.route({ query: "那个东西怎么样了" });
  const again = await engine2.route({ query: "那个东西怎么样了" });
  assert.equal(again.cacheHit, false);
  assert.equal(uncertainCloud.calls.length, 2);
});

// ------------------------------------------------------------------- audit

test("audit stores a query hash, never the raw text", async () => {
  const lines = [];
  const cloud = fakeCloud();
  const engine = makeEngine({
    config: routerConfig({
      cloud: { enabled: true },
      audit: { enabled: true, store_query: "hash" },
    }),
    cloud,
    auditLines: lines,
  });
  await engine.route({
    query: "上次那个蓝色马克杯",
    sessionId: "s-1",
    turn: 1,
  });
  const audit = engine._audit;
  await audit.flush();
  assert.equal(lines.length, 1);
  const record = JSON.parse(lines[0]);
  assert.equal(record.query, undefined);
  assert.equal(record.query_hash, hashQuery("上次那个蓝色马克杯"));
  assert.equal(record.query_chars, 9);
  assert.equal(record.decision, "retrieve");
  assert.equal(record.source, "cloud");
  assert.equal(record.session_id, "s-1");
  assert.equal(record.rules_version, RULES_VERSION);
  assert.equal(lines[0].includes("蓝色马克杯"), false);
});

test("shadow mode marks every decision without changing the decision", async () => {
  const engine = makeEngine({
    config: routerConfig({ shadow_mode: true }),
  });
  const result = await engine.route({ query: "你好" });
  assert.equal(result.shadow, true);
  assert.equal(result.decision, "skip");
  const status = await engine.status();
  assert.equal(status.shadowMode, true);
});

test("status reports budget, circuit, and stats", async () => {
  const cloud = fakeCloud();
  const engine = makeEngine({
    config: routerConfig({ cloud: { enabled: true } }),
    cloud,
  });
  await engine.route({ query: "上次那个方案" });
  const status = await engine.status();
  assert.equal(status.enabled, true);
  assert.equal(status.mode, "auto");
  assert.equal(status.budgetMs, 200);
  assert.equal(status.cloud.configured, true);
  assert.equal(status.regex.rulesVersion, RULES_VERSION);
  assert.equal(status.stats.turns, 1);
  assert.equal(status.stats.retrieveRate, 1);
});

test("config normalization clamps the budget and keeps neutral defaults", () => {
  const config = normalizeRouterConfig({ budget_ms: 9999 });
  assert.equal(config.budgetMs, 200);
  assert.equal(config.timeoutPolicy, "retrieve");
  assert.equal(config.onInternalError, "retrieve");
  assert.equal(config.regex.defaultDecision, "retrieve");
  const low = normalizeRouterConfig({ budget_ms: 1 });
  assert.equal(low.budgetMs, 50);
  assert.equal(normalizeRouterConfig({ shadow_mode: true }).shadowMode, true);
});

// ------------------------------------------------------- pre_retrieval wiring

function fakeCtx() {
  const handlers = {};
  return {
    handlers,
    on: (name, handler) => {
      handlers[name] = handler;
      return () => {};
    },
    effect: (factory) => {
      const dispose = factory();
      return dispose;
    },
  };
}

function hintResult(text = "remembered fact") {
  return {
    ok: true,
    result: {
      results: [
        {
          content: text,
          source_type: "user_message",
          similarity: 0.9,
          metadata: { timestamp_ms: Date.now() },
        },
      ],
    },
  };
}

function preStep(handler, query, signal) {
  const decision = {
    kind: "continue",
    messages: [{ role: "assistant", content: [{ type: "text", text: "ok" }] }],
  };
  return handler(
    {
      messages: [{ role: "user", content: [{ type: "text", text: query }] }],
      step: 1,
      signal,
    },
    async () => decision,
  );
}

function registerRouter({ config, router, queryMemory, calls }) {
  const ctx = fakeCtx();
  registerPreRetrieval(ctx, {
    queryMemory: async (...args) => {
      calls.push(args[0]);
      return queryMemory(...args);
    },
    holoLog: () => {},
    diag: () => {},
    getConfig: () => config,
    router,
    createUserMessage: (options) => ({ role: "user", ...options }),
  });
  return ctx.handlers["agent/pre-step"];
}

test("router.enabled=false restores the legacy behaviour", async () => {
  const calls = [];
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: false } },
    router: null,
    queryMemory: () => hintResult(),
    calls,
  });
  const short = await preStep(handler, "你好");
  assert.equal(short.messages.length, 1);
  assert.equal(calls.length, 0);

  const long = await preStep(handler, "帮我检查一下上次那个项目的进度");
  assert.equal(calls.length, 1);
  assert.equal(long.messages.length, 2);
  assert.equal(long.messages[0].source.kind, "memory-recall");
});

test("router skip prevents retrieval; retrieve performs it", async () => {
  const calls = [];
  const skipHandler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: { route: async () => ({ decision: "skip", source: "local", shadow: false }) },
    queryMemory: () => hintResult(),
    calls,
  });
  const skipped = await preStep(skipHandler, "帮我写一个快速排序函数");
  assert.equal(calls.length, 0);
  assert.equal(skipped.messages.length, 1);

  const retrieveHandler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: { route: async () => ({ decision: "retrieve", source: "local", shadow: false }) },
    queryMemory: () => hintResult(),
    calls,
  });
  const retrieved = await preStep(retrieveHandler, "上次那个蓝色马克杯");
  assert.equal(calls.length, 1);
  assert.equal(retrieved.messages.length, 2);
});

test("shadow mode retrieves exactly like the legacy path", async () => {
  const calls = [];
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: {
      route: async () => ({ decision: "skip", source: "regex", shadow: true }),
    },
    queryMemory: () => hintResult(),
    calls,
  });
  const decision = await preStep(handler, "帮我写一个快速排序函数");
  assert.equal(calls.length, 1);
  assert.equal(decision.messages.length, 2);
});

test("aborted pre-step never retrieves", async () => {
  const calls = [];
  const controller = new AbortController();
  controller.abort();
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: {
      route: async () => ({
        decision: "retrieve",
        source: "regex",
        cancelled: true,
        shadow: false,
      }),
    },
    queryMemory: () => hintResult(),
    calls,
  });
  const decision = await preStep(handler, "上次那个方案", controller.signal);
  assert.equal(calls.length, 0);
  assert.equal(decision.messages.length, 1);
});

test("a thrown router defaults to retrieve (engine contract breach)", async () => {
  const calls = [];
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: {
      route: async () => {
        throw new Error("router exploded");
      },
    },
    queryMemory: () => hintResult(),
    calls,
  });
  await preStep(handler, "上次那个方案");
  assert.equal(calls.length, 1);
});

test("end-to-end: engine + pre_retrieval retrieval gate", async () => {
  const calls = [];
  const local = fakeLocal({
    impl: () => localRoute({ decision: "skip", probabilityRetrieve: 0.05 }),
  });
  const engine = makeEngine({ config: routerConfig(), local });
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: engine,
    queryMemory: () => hintResult(),
    calls,
  });
  const skipped = await preStep(handler, "帮我写一个快速排序函数");
  assert.equal(calls.length, 0);
  assert.equal(skipped.messages.length, 1);

  const local2 = fakeLocal({ impl: () => localRoute({ probabilityRetrieve: 0.93 }) });
  const engine2 = makeEngine({ config: routerConfig(), local: local2 });
  const handler2 = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: engine2,
    queryMemory: () => hintResult(),
    calls,
  });
  const retrieved = await preStep(handler2, "上次那个蓝色马克杯");
  assert.equal(calls.length, 1);
  assert.equal(retrieved.messages.length, 2);
});

// -------------------------------------------------------------------- runner

const originalLog = console.log;
console.log = () => {};
let passed = 0;
try {
  for (const [name, fn] of tests) {
    try {
      await fn();
    } catch (error) {
      console.error(`\nFAIL: ${name}`);
      throw error;
    }
    passed += 1;
  }
} finally {
  console.log = originalLog;
}
console.log(`router_engine.test.mjs: ${passed} tests passed`);
