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
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  CircuitBreaker,
  RouteCache,
  RouterAudit,
  createRouterEngine,
  hashQuery,
  normalizeRouterConfig,
  resolveAuditPath,
} from "../dsh_plugin/router_engine.js";
import {
  EXPLICIT,
  HARD_SKIP,
  NEUTRAL,
  RULES_VERSION,
  WEAK,
  guard,
  normalizeQuery,
} from "../dsh_plugin/router_regex.js";
import { CloudRouter, CloudRouterError } from "../dsh_plugin/router_cloud.js";
import { LocalRouterClient } from "../dsh_plugin/router_local.js";
import {
  createMemoryHint,
  recallDecision,
  registerPreRetrieval,
  topSimilarity,
} from "../dsh_plugin/pre_retrieval.js";

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
  // Weak cues must not decide anything by themselves: "现在", "那个东西",
  // bare "记得", "变化" are handed to the local BGE head.
  assert.equal(guard("现在几点了").kind, WEAK);
  assert.equal(guard("那个东西怎么样了").kind, WEAK);
  assert.equal(guard("记得带伞").kind, WEAK);
  assert.equal(guard("这个方案有什么变化").kind, WEAK);
  assert.equal(guard("你还记得我的偏好吗").kind, EXPLICIT);
  assert.equal(guard("帮我写一个快速排序函数").kind, NEUTRAL);
  assert.equal(normalizeQuery("  你  好  "), "你 好");
});

test("weak cues go to the local judge instead of being hard-retrieved", async () => {
  const local = fakeLocal({
    impl: () => localRoute({ decision: "skip", probabilityRetrieve: 0.03 }),
  });
  const engine = makeEngine({ config: routerConfig(), local });
  const result = await engine.route({ query: "现在几点了？那个东西怎么样了" });
  assert.equal(local.calls.length, 1);
  assert.equal(result.source, "local");
  assert.equal(result.decision, "skip");
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

const TEST_TMP = fileURLToPath(new URL("../.test-tmp/", import.meta.url));

function auditDir(label) {
  const dir = join(
    TEST_TMP,
    `audit-${label}-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
  );
  mkdirSync(dir, { recursive: true });
  return dir;
}

test("audit path is the documented ~/.dsh/dream/router.jsonl", () => {
  const path = resolveAuditPath({ env: {}, home: "C:\\Users\\tester" });
  assert.equal(
    path,
    resolve("C:\\Users\\tester", ".dsh", "dream", "router.jsonl"),
  );
  const override = resolveAuditPath({
    env: { DSH_DREAM_ROUTER_AUDIT: "D:\\logs\\router.jsonl" },
    home: "C:\\Users\\tester",
  });
  assert.equal(override, resolve("D:\\logs\\router.jsonl"));
  // A DSH profile directory must not move the documented path any more.
  const withProfile = resolveAuditPath({
    env: { DSH_PROFILE_DIR: "C:\\Users\\tester\\.dsh\\profiles\\desktop" },
    home: "C:\\Users\\tester",
  });
  assert.equal(
    withProfile,
    resolve("C:\\Users\\tester", ".dsh", "dream", "router.jsonl"),
  );
  // `DSH_HOME` already points at the `.dsh` directory itself.
  const withDshHome = resolveAuditPath({
    env: { DSH_HOME: "E:\\dsh-home" },
    home: "C:\\Users\\tester",
  });
  assert.equal(withDshHome, resolve("E:\\dsh-home", "dream", "router.jsonl"));
});

test("ensureFile creates the parent directory and an empty log file", async () => {
  const dir = auditDir("ensure");
  const file = join(dir, "nested", "router.jsonl");
  try {
    assert.equal(existsSync(file), false);
    const audit = new RouterAudit({ path: file });
    const result = await audit.ensureFile();
    assert.equal(result.ok, true);
    assert.equal(result.path, file);
    assert.equal(existsSync(file), true);
    assert.equal(statSync(file).size, 0);
    assert.equal(audit.describe().fileCreated, true);
    // Idempotent: a second call must not truncate an existing log.
    await audit.record({
      query: "上次那个方案",
      traceId: "t1",
      decision: "retrieve",
      source: "regex",
      probabilityRetrieve: 1,
      confidence: 1,
      threshold: 0.55,
      uncertaintyBand: 0.1,
      latencyMs: 0.2,
      budgetMs: 200,
    });
    await audit.flush();
    const before = statSync(file).size;
    await audit.ensureFile();
    assert.equal(statSync(file).size, before);
    assert.equal(audit.describe().writes, 1);
    rmSync(dir, { recursive: true, force: true });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("engine warmup creates the audit file and the first route appends hashed JSONL", async () => {
  const dir = auditDir("engine");
  const file = join(dir, "router.jsonl");
  try {
    const engine = createRouterEngine({
      getConfig: () => ({
        router: routerConfig({
          audit: { enabled: true, store_query: "hash" },
        }),
      }),
      local: fakeLocal(),
      auditPath: file,
      diag: () => {},
    });
    await engine.warmup();
    assert.equal(existsSync(file), true, "warmup must create the audit file");
    const status = await engine.status();
    assert.equal(status.audit.enabled, true);
    assert.equal(status.audit.path, file);
    assert.equal(status.audit.fileCreated, true);

    await engine.route({ query: "上次那个蓝色马克杯", sessionId: "s-9", turn: 1 });
    await engine._audit.flush();
    const lines = readFileSync(file, "utf8").trim().split("\n");
    assert.equal(lines.length, 1);
    const record = JSON.parse(lines[0]);
    assert.equal(record.query, undefined);
    assert.equal(record.query_hash, hashQuery("上次那个蓝色马克杯"));
    assert.equal(record.decision, "retrieve");
    const after = await engine.status();
    assert.equal(after.audit.writes, 1);
    assert.equal(after.audit.lastError, null);
    rmSync(dir, { recursive: true, force: true });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("hard-skip guard decisions are audited too", async () => {
  const dir = auditDir("skip");
  const file = join(dir, "router.jsonl");
  try {
    const engine = createRouterEngine({
      getConfig: () => ({
        router: routerConfig({
          audit: { enabled: true, store_query: "hash" },
        }),
      }),
      auditPath: file,
      diag: () => {},
    });
    await engine.warmup();
    const cases = [
      ["你好", "greeting"],
      ["/clear", "command"],
      ["别翻旧账", "suppression"],
      ["   ", "empty"],
    ];
    for (const [query, reason] of cases) {
      const result = await engine.route({ query });
      assert.equal(result.decision, "skip");
      assert.equal(result.source, "regex");
      assert.equal(result.reason, reason);
    }
    await engine._audit.flush();
    const records = readFileSync(file, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    assert.equal(records.length, 4);
    assert.deepEqual(
      records.map((record) => record.reason),
      ["greeting", "command", "suppression", "empty"],
    );
    for (const record of records) {
      assert.equal(record.decision, "skip");
      assert.equal(record.source, "regex");
      assert.equal(record.probability, 0);
      assert.equal(record.query, undefined);
      assert.match(record.query_hash, /^[0-9a-f]{64}$/);
    }
    rmSync(dir, { recursive: true, force: true });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("audit failures are recorded instead of thrown", async () => {
  const dir = auditDir("broken");
  try {
    const blocker = join(dir, "not-a-directory");
    writeFileSync(blocker, "x", "utf8");
    const audit = new RouterAudit({ path: join(blocker, "router.jsonl") });
    const result = await audit.ensureFile();
    assert.equal(result.ok, false);
    assert.ok(audit.describe().lastError, "lastError should explain the failure");
    // record() must stay non-throwing; the decision path cannot fail on audit.
    audit.record({
      query: "x",
      traceId: "t2",
      decision: "skip",
      source: "regex",
      probabilityRetrieve: 0,
      confidence: 1,
      threshold: 0.55,
      uncertaintyBand: 0.1,
      latencyMs: 0,
      budgetMs: 200,
    });
    await audit.flush();
    assert.equal(audit.describe().writes, 0);
    rmSync(dir, { recursive: true, force: true });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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

// ------------------------------------------------------------- cloud adapter

test("custom cloud endpoint fills preset path/model and sends only the query", async () => {
  const calls = [];
  const router = new CloudRouter({
    getConfig: () => ({
      router: {
        threshold: 0.55,
        uncertainty_band: 0.1,
        cloud: {
          enabled: true,
          active: "custom",
          providers: {
            custom: {
              base_url: "https://ws-x.example.com",
              path: "",
              model: "",
              api_key_env: "",
            },
          },
        },
      },
    }),
    resolveCredential: async () => "",
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      return {
        ok: true,
        status: 200,
        headers: { get: () => null },
        json: async () => ({ answers: { need_rag: { noul: 0.93 } } }),
      };
    },
  });
  const provider = router.providerConfig();
  assert.equal(provider.path, "/compatible-mode/v1/systemone");
  assert.equal(provider.model, "decision-model-preview");
  const result = await router.route(
    { query: "上次那个蓝色马克杯在哪" },
    { deadlineMs: 150 },
  );
  assert.equal(
    calls[0].url,
    "https://ws-x.example.com/compatible-mode/v1/systemone",
  );
  assert.equal(result.decision, "retrieve");
  assert.equal(result.uncertain, false);
  const body = JSON.parse(calls[0].init.body);
  assert.equal(body.state, "上次那个蓝色马克杯在哪");
  assert.equal(Object.keys(body.questions).length, 1);
  assert.equal(body.questions.need_rag.type, "noul");
  assert.equal(calls[0].init.headers.authorization, undefined);
});

test("cloud status codes map to stable error codes with cooldowns", async () => {
  const makeRouter = (status, headers = { get: () => null }) =>
    new CloudRouter({
      getConfig: () => ({
        router: {
          cloud: {
            enabled: true,
            active: "custom",
            providers: { custom: { base_url: "https://ws-x.example.com" } },
          },
        },
      }),
      resolveCredential: async () => "key",
      fetchImpl: async () => ({ ok: false, status, headers }),
    });
  await assert.rejects(
    makeRouter(401).route({ query: "上次说的那个" }, { deadlineMs: 150 }),
    (error) => error instanceof CloudRouterError && error.code === "CLOUD_AUTH",
  );
  await assert.rejects(
    makeRouter(451).route({ query: "上次说的那个" }, { deadlineMs: 150 }),
    (error) => error.code === "CLOUD_REGION",
  );
  await assert.rejects(
    makeRouter(429, { get: () => "2" }).route(
      { query: "上次说的那个" },
      { deadlineMs: 150 },
    ),
    (error) =>
      error.code === "CLOUD_RATE_LIMIT" && error.retryAfterMs === 2000,
  );
  await assert.rejects(
    makeRouter(500).route({ query: "上次说的那个" }, { deadlineMs: 150 }),
    (error) => error.code === "CLOUD_SERVER",
  );
});

test("a response without a need_rag answer is a bad response", async () => {
  const router = new CloudRouter({
    getConfig: () => ({
      router: {
        cloud: {
          enabled: true,
          active: "custom",
          providers: { custom: { base_url: "https://ws-x.example.com" } },
        },
      },
    }),
    resolveCredential: async () => "",
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      headers: { get: () => null },
      json: async () => ({ answers: {} }),
    }),
  });
  await assert.rejects(
    router.route({ query: "上次说的那个" }, { deadlineMs: 150 }),
    (error) => error.code === "CLOUD_BAD_RESPONSE",
  );
});

test("missing credentials keep the cloud layer out", async () => {
  let fetched = 0;
  const router = new CloudRouter({
    getConfig: () => ({
      router: {
        cloud: {
          enabled: true,
          active: "custom",
          providers: {
            custom: {
              base_url: "https://ws-x.example.com",
              api_key_env: "MISSING_KEY",
            },
          },
        },
      },
    }),
    resolveCredential: async () => "",
    fetchImpl: async () => {
      fetched += 1;
      return { ok: true, status: 200, json: async () => ({}) };
    },
  });
  await assert.rejects(
    router.route({ query: "上次说的那个" }, { deadlineMs: 150 }),
    (error) => error.code === "CLOUD_NOT_CONFIGURED",
  );
  assert.equal(fetched, 0);
});

// ------------------------------------------------------------- local client

test("local client keeps retrying while warming and adopts ready", async () => {
  const responses = [
    { ok: false, code: "LOCAL_NOT_READY", state: "warming" },
    {
      ok: true,
      backend: "bge-head",
      model_id: "m:head:v1",
      probability: 0.91,
      decision: "retrieve",
      confidence: 0.91,
      uncertain: false,
      threshold: 0.55,
      uncertainty_band: 0.1,
      latency_ms: 2.0,
    },
  ];
  const bridge = {
    isBusy: () => false,
    routeQuery: async () => responses.shift(),
    routerStatus: async () => ({ ok: true, ready: false, state: "warming" }),
  };
  const client = new LocalRouterClient({
    bridge,
    getConfig: () => ({ router: { local: { backend: "bge-head" } } }),
    diag: () => {},
  });
  assert.equal(client.isAvailable(), true);
  await assert.rejects(
    client.route({ query: "上次那个方案" }, { deadlineMs: 100 }),
    (error) => error.code === "LOCAL_NOT_READY",
  );
  // A warming backend must stay eligible: the old cached-ready gate pinned the
  // router to regex forever after one cold turn.
  assert.equal(client.isAvailable(), true);
  assert.equal(client.ready, false);
  const result = await client.route(
    { query: "上次那个方案" },
    { deadlineMs: 100 },
  );
  assert.equal(result.decision, "retrieve");
  assert.equal(client.ready, true);
  assert.equal(client.state, "ready");
});

test("local client latches off only on a model mismatch", async () => {
  const bridge = {
    isBusy: () => false,
    routeQuery: async () => ({ ok: false, code: "LOCAL_MODEL_MISMATCH" }),
    routerStatus: async () => ({ ok: true, ready: false, state: "warming" }),
  };
  const client = new LocalRouterClient({
    bridge,
    getConfig: () => ({ router: { local: { backend: "bge-head" } } }),
    diag: () => {},
  });
  await assert.rejects(
    client.route({ query: "x" }, { deadlineMs: 100 }),
    (error) => error.code === "LOCAL_MODEL_MISMATCH",
  );
  assert.equal(client.isAvailable(), false);
  assert.equal(client.state, "failed");
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

function hintResult(text = "remembered fact", overrides = {}) {
  return {
    ok: true,
    result: {
      results: [
        {
          content: text,
          source_type: "user_message",
          similarity: 0.9,
          metadata: { timestamp_ms: Date.now() },
          ...overrides,
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

test("recall gate accepts fused hybrids and keyword hits", () => {
  const hybrid = recallDecision([
    { similarity: 0.59, retrieval_source: "hybrid", rrf_score: 0.03 },
  ]);
  assert.equal(hybrid.inject, true);
  assert.equal(hybrid.reason, "hybrid");

  const vectorLow = recallDecision([
    { similarity: 0.55, retrieval_source: "vector", rrf_score: 0.01 },
  ]);
  assert.equal(vectorLow.inject, false);
  assert.equal(vectorLow.reason, "below_threshold");

  const keyword = recallDecision([
    { similarity: null, retrieval_source: "bm25", rrf_score: 0.016 },
  ]);
  assert.equal(keyword.inject, true);
  assert.equal(keyword.reason, "keyword");

  const anyHit = recallDecision([
    { similarity: 0.3, retrieval_source: "hybrid", rrf_score: 0.03 },
    { similarity: 0.7, retrieval_source: "vector", rrf_score: 0.01 },
  ]);
  assert.equal(anyHit.inject, true);
  assert.equal(anyHit.reason, "vector_any_hit");

  assert.equal(
    topSimilarity([{ similarity: 0.3 }, { distance: 0.2 }]).toFixed(2),
    "0.80",
  );
  assert.equal(recallDecision([]).reason, "no_hits");
});

test("memory hint tells the model to trust confirmed facts", () => {
  const hint = createMemoryHint({
    results: [
      {
        content: "我的东山精密成本是92元，浇水闹钟已取消",
        source_type: "user_message",
        metadata: { timestamp_ms: Date.UTC(2026, 10, 10) },
      },
    ],
  });
  assert.ok(hint.includes("<memory-hint>"));
  assert.ok(hint.includes("优先直接使用"));
  assert.ok(hint.includes("不要为了确认这些事实去读取文件"));
  assert.ok(hint.includes("浇水闹钟已取消"));
});

test("integration: a hybrid 0.59 recall still injects a hint", async () => {
  const calls = [];
  const handler = registerRouter({
    config: { enable_auto_recall: true, router: { enabled: true } },
    router: {
      route: async () => ({ decision: "retrieve", source: "local", shadow: false }),
    },
    queryMemory: () =>
      hintResult("我的东山精密成本是92元，浇水闹钟已取消", {
        similarity: 0.59,
        retrieval_source: "hybrid",
        rrf_score: 0.0328,
      }),
    calls,
  });
  const decision = await preStep(handler, "浇水闹钟取消了吗");
  assert.equal(calls.length, 1);
  assert.equal(decision.messages.length, 2);
  assert.ok(
    decision.messages[0].content[0].text.includes("优先直接使用"),
  );
});

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
