/**
 * Config-bridge mapping tests.
 *
 * Run with:  node tests/config_bridge.test.mjs
 *
 * The settings page posts flat `router_*` fields; the bridge file stores a
 * nested `router` block. These tests pin that mapping (including preserving
 * advanced settings the page never renders) without needing a DSH runtime.
 */

import assert from "node:assert/strict";
import {
  normalizePayload,
  normalizeRouterPayload,
  redactPayload,
} from "../dsh_plugin/config_bridge.js";

const tests = [];
function test(name, fn) {
  tests.push([name, fn]);
}

const previous = {
  enable_memory: true,
  enable_auto_recall: true,
  rag_path: "C:/data/chroma",
  python_path: "",
  hf_home_path: "",
  max_window_tokens: 1500,
  retrieval_top_k: 5,
  embedding_provider: "local",
  embedding_model: "",
  embedding_api_key: "",
  embedding_base_url: "",
  auto_rebuild_collection: true,
  holographic_mode: false,
  router: {
    enabled: false,
    mode: "regex",
    budget_ms: 180,
    threshold: 0.62,
    uncertainty_band: 0.08,
    timeout_policy: "bypass_main",
    error_policy: "fallback_regex",
    on_internal_error: "retrieve",
    shadow_mode: true,
    regex: { enabled: true, default_decision: "retrieve", explicit_fast_path: true },
    local: { backend: "off", hedge: false, model_dir: "", warmup_timeout_ms: 20000 },
    cloud: {
      enabled: true,
      active: "ali-bailian",
      budget_ms: 120,
      providers: {
        "ali-bailian": {
          base_url: "https://old.example.com",
          path: "/old/path",
          model: "old-model",
          api_key_env: "DASHSCOPE_API_KEY",
        },
      },
    },
    cache: { enabled: true, ttl_ms: 60000, max_entries: 64 },
    circuit: { failure_threshold: 5, cooldown_ms: 60000 },
    audit: { enabled: true, store_query: "full", max_bytes: 1048576 },
  },
};

function clientPayload(overrides = {}) {
  return {
    enable_memory: true,
    enable_auto_recall: true,
    rag_path: "C:/data/chroma",
    python_path: "",
    hf_home_path: "D:/models",
    max_window_tokens: 2000,
    retrieval_top_k: 6,
    embedding_provider: "local",
    embedding_model: "",
    embedding_api_key: "",
    embedding_base_url: "",
    auto_rebuild_collection: true,
    holographic_mode: false,
    router_enabled: true,
    router_mode: "cloud",
    router_cloud_enabled: true,
    router_cloud_provider: "custom",
    router_cloud_base_url: "https://ws-abc.cn-beijing.maas.aliyuncs.com",
    router_cloud_path: "/compatible-mode/v1/systemone",
    router_cloud_model: "decision-model-preview",
    router_cloud_api_key_env: "DASHSCOPE_API_KEY",
    ...overrides,
  };
}

test("custom cloud endpoint maps into the nested router block", () => {
  const payload = normalizePayload(clientPayload(), previous);
  const router = payload.router;
  assert.equal(router.enabled, true);
  assert.equal(router.mode, "cloud");
  assert.equal(router.cloud.enabled, true);
  assert.equal(router.cloud.active, "custom");
  assert.equal(
    router.cloud.providers.custom.base_url,
    "https://ws-abc.cn-beijing.maas.aliyuncs.com",
  );
  assert.equal(
    router.cloud.providers.custom.path,
    "/compatible-mode/v1/systemone",
  );
  assert.equal(
    router.cloud.providers.custom.model,
    "decision-model-preview",
  );
  assert.equal(
    router.cloud.providers.custom.api_key_env,
    "DASHSCOPE_API_KEY",
  );
  // The page did not render these, so the previous values must survive.
  assert.equal(router.budget_ms, 180);
  assert.equal(router.threshold, 0.62);
  assert.equal(router.uncertainty_band, 0.08);
  assert.equal(router.timeout_policy, "bypass_main");
  assert.equal(router.shadow_mode, true);
  assert.equal(router.regex.explicit_fast_path, true);
  assert.equal(router.local.backend, "off");
  assert.equal(router.local.hedge, false);
  assert.equal(router.cache.ttl_ms, 60000);
  assert.equal(router.circuit.failure_threshold, 5);
  assert.equal(router.audit.store_query, "hash"); // always hash-only on disk
  // Non-router fields still map as before.
  assert.equal(payload.hf_home_path, "D:/models");
  assert.equal(payload.max_window_tokens, 2000);
  assert.equal(payload.retrieval_top_k, 6);
});

test("clearing the endpoint disables the cloud layer without touching the rest", () => {
  const payload = normalizePayload(
    clientPayload({
      router_cloud_enabled: false,
      router_cloud_provider: "",
      router_cloud_base_url: "",
      router_cloud_path: "",
      router_cloud_model: "",
      router_cloud_api_key_env: "",
    }),
    previous,
  );
  assert.equal(payload.router.cloud.enabled, false);
  assert.equal(payload.router.enabled, true);
  assert.equal(payload.router.mode, "cloud");
  assert.equal(payload.router.budget_ms, 180);
  assert.equal(payload.router.shadow_mode, true);
});

test("a payload without router fields preserves the whole previous router block", () => {
  const legacy = { enable_memory: true, holographic_mode: true };
  const payload = normalizePayload(legacy, previous);
  assert.equal(payload.router.enabled, false);
  assert.equal(payload.router.mode, "regex");
  assert.equal(payload.router.local.backend, "off");
  assert.equal(payload.router.cloud.active, "ali-bailian");
  assert.equal(
    payload.router.cloud.providers["ali-bailian"].base_url,
    "https://old.example.com",
  );
  assert.equal(payload.holographic_mode, true);
});

test("flat settings-form fields override the nested block", () => {
  const payload = normalizePayload(
    {
      enable_memory: false,
      router_enabled: false,
      router_mode: "regex",
      router_local_backend: "bge-head",
    },
    previous,
  );
  assert.equal(payload.enable_memory, false);
  assert.equal(payload.router.enabled, false);
  assert.equal(payload.router.mode, "regex");
  assert.equal(payload.router.local.backend, "bge-head");
  assert.equal(payload.router.budget_ms, 180); // untouched
});

test("fresh install falls back to safe router defaults", () => {
  const router = normalizeRouterPayload({});
  assert.equal(router.enabled, true);
  assert.equal(router.mode, "auto");
  assert.equal(router.budget_ms, 200);
  assert.equal(router.threshold, 0.55);
  assert.equal(router.uncertainty_band, 0.1);
  assert.equal(router.timeout_policy, "retrieve");
  assert.equal(router.on_internal_error, "retrieve");
  assert.equal(router.shadow_mode, false);
  assert.equal(router.regex.default_decision, "retrieve");
  assert.equal(router.local.backend, "bge-head");
  assert.equal(router.local.hedge, true);
  assert.equal(router.cloud.enabled, false);
  assert.equal(router.cloud.providers && typeof router.cloud.providers, "object");
  assert.equal(router.audit.store_query, "hash");
});

test("volatile reference values are unwrapped", () => {
  const ref = (value) => ({ get: () => value });
  const payload = normalizePayload(
    {
      enable_memory: ref(false),
      enable_auto_recall: ref(true),
      rag_path: ref("D:/chroma"),
      max_window_tokens: ref(2048),
      router_enabled: ref(false),
      router_mode: ref("local"),
    },
    previous,
  );
  assert.equal(payload.enable_memory, false);
  assert.equal(payload.rag_path, "D:/chroma");
  assert.equal(payload.max_window_tokens, 2048);
  assert.equal(payload.router.enabled, false);
  assert.equal(payload.router.mode, "local");
});

test("redactPayload never leaks the embedding key", () => {
  const redacted = redactPayload({
    embedding_api_key: "sk-secret",
    enable_memory: true,
  });
  assert.equal(redacted.embedding_api_key, "***");
  assert.equal(redacted.enable_memory, true);
  assert.equal(redactPayload({}).embedding_api_key, "");
});

let passed = 0;
for (const [name, fn] of tests) {
  try {
    await fn();
  } catch (error) {
    console.error(`\nFAIL: ${name}`);
    throw error;
  }
  passed += 1;
}
console.log(`config_bridge.test.mjs: ${passed} tests passed`);
