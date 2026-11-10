/**
 * Client settings-page contract tests (no DSH runtime, no browser).
 *
 * `client.js` is a plain-JS Client plugin: it calls
 * `window.__ModuleLoader__.load({ factory })` at load time, and the factory
 * only builds data + component functions. This test stubs the module loader and
 * React just enough to execute the factory and `apply(ctx)`, then checks:
 *
 * - every `tr('key')` used by the page exists in both zh and en dictionaries;
 * - the Router section, the advanced collapsible block, and the plain-language
 *   copy the task requires are present;
 * - the save path posts the flat `router_*` fields the Host maps into the
 *   nested bridge config.
 *
 * Run with:  node tests/client_ui.test.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const clientPath = fileURLToPath(
  new URL("../dsh_plugin/client.js", import.meta.url),
);
const source = readFileSync(clientPath, "utf8");

const fakeReact = {
  Fragment: Symbol("Fragment"),
  createElement: () => null,
  useState: () => [null, () => {}],
  useEffect: () => {},
  useLayoutEffect: () => {},
  useCallback: (fn) => fn,
  useRef: () => ({ current: null }),
  useSyncExternalStore: () => ({ status: "ready", value: {} }),
};

let moduleEntry = null;
globalThis.window = {
  __ModuleLoader__: {
    load: (entry) => {
      moduleEntry = entry;
    },
  },
};

// Execute the top-level registration; the factory itself is not invoked yet.
new Function("window", "React", source)(globalThis.window, fakeReact);

if (!moduleEntry || typeof moduleEntry.factory !== "function") {
  throw new Error("client.js did not register a module factory");
}

const exported = moduleEntry.factory((name) => {
  if (name === "react") return fakeReact;
  throw new Error(`unexpected require(${name})`);
});

assert.equal(Array.isArray(exported.inject), true);
assert.equal(typeof exported.apply, "function");

const registered = {};
const slotRegistrations = [];
let section = null;
const ctx = {
  effect: (fn) => fn(),
  locale: {
    bind: () => (key) => key,
    register: (ns, dictionaries) => {
      registered[ns] = dictionaries;
      return () => {};
    },
  },
  configForms: {
    get: () => ({}),
    whileServed: (_ids, callback) => callback(),
  },
  slots: {
    inject: (_name, callback) => callback(),
    register: (meta, component) => {
      slotRegistrations.push({ meta, component });
      if (meta.name === "settings.section") section = { meta, component };
      return () => {};
    },
  },
};
exported.apply(ctx);

const dictionaries = registered["settings.agentMemory"];
assert.ok(dictionaries, "locale.register(settings.agentMemory) was not called");
assert.ok(dictionaries.zh && dictionaries.en, "zh/en dictionaries missing");
assert.ok(section, "settings.section was not registered");
assert.equal(section.meta.id, "DSH-DREAM");

const indicatorRegistration = slotRegistrations.find(
  (entry) => entry.meta.name === "conversation.input.dock",
);
assert.ok(
  indicatorRegistration,
  "conversation.input.dock indicator was not registered",
);
assert.equal(indicatorRegistration.meta.id, "dsh-dream-holo-indicator");
assert.equal(
  indicatorRegistration.meta.locale,
  "settings.agentMemory",
  "indicator must reuse the existing zh/en dictionary",
);
assert.equal(typeof indicatorRegistration.component, "function");

const usedKeys = new Set();
for (const match of source.matchAll(/tr\('([^']+)'\)/g)) usedKeys.add(match[1]);
usedKeys.add("nav"); // used through `t('nav')` in apply()
const missingZh = [...usedKeys].filter((key) => !(key in dictionaries.zh));
const missingEn = [...usedKeys].filter((key) => !(key in dictionaries.en));
assert.deepEqual(missingZh, [], `missing zh keys: ${missingZh.join(", ")}`);
assert.deepEqual(missingEn, [], `missing en keys: ${missingEn.join(", ")}`);

// --- Holographic indicator copy / payload --------------------------------
assert.equal(dictionaries.zh.holoIndicatorOn, "全息 · 落库中");
assert.equal(dictionaries.en.holoIndicatorOn, "HOLO · RECORDING");
assert.equal(dictionaries.zh.holoStyleBadge, "badge（小灯）");
assert.equal(dictionaries.zh.holoStyleTheme, "theme（白底黑字）");
assert.match(dictionaries.zh.holoIndicatorHint, /不是全息开关/);
assert.match(dictionaries.en.holoIndicatorHint, /not the holographic switch/i);
assert.ok(
  source.includes("holo_indicator: {"),
  "save payload is missing holo_indicator",
);
assert.ok(
  source.includes("path: ['holo_indicator']"),
  "settings mutation is missing holo_indicator",
);
assert.ok(
  source.includes('.dsh-dream-holo-theme'),
  "theme class is missing",
);
assert.ok(
  source.includes('[data-conversation-region="chat"]'),
  "theme is not scoped to the chat container",
);

// --- Required plain-language copy ---------------------------------------
assert.equal(dictionaries.zh.routerTitle, "智能路由（Router）");
assert.equal(dictionaries.zh.routerEnabled, "智能意图截断");
assert.equal(
  dictionaries.zh.routerEnabledHint,
  "开启后，AI 会先判断话题是否需要翻阅旧记忆。闲聊直接回复，不浪费时间查库。",
);
assert.equal(dictionaries.zh.routerMode, "判断优先级");
assert.equal(
  dictionaries.zh.routerModeHint,
  "默认采用零配置的本地极速判断（约 5 毫秒，不需下载新模型）。如果遇到拿不准的话题，默认放行去查库，宁可多查也不失忆。",
);
assert.equal(dictionaries.zh.routerModeAuto, "auto（自动-默认）");
assert.equal(dictionaries.zh.routerModeCloud, "cloud（优先云端）");
assert.equal(dictionaries.zh.routerModeLocal, "local（仅本地极速）");
assert.equal(dictionaries.zh.routerModeRegex, "regex（仅关键词规则）");
assert.equal(
  dictionaries.zh.routerAdvancedTitle,
  "高级：自定义外接判断模型（选填）",
);
assert.equal(
  dictionaries.zh.routerAdvancedNotice,
  "普通用户请留空！留空即代表使用内置极速本地判断。如果你是进阶用户，可在此接入外部云端模型（如阿里百炼、Jev）来提升复杂语境的判断准确率。",
);
assert.equal(dictionaries.zh.routerCloudBaseUrl, "云端接口地址");
assert.equal(
  dictionaries.zh.routerCloudBaseUrlPlaceholder,
  "例如 https://ws-xxx.cn-beijing.maas.aliyuncs.com",
);
assert.equal(dictionaries.zh.routerCloudPath, "请求路径");
assert.equal(
  dictionaries.zh.routerCloudPathPlaceholder,
  "默认 /compatible-mode/v1/systemone",
);
assert.equal(dictionaries.zh.routerCloudModel, "云端模型名称");
assert.equal(
  dictionaries.zh.routerCloudModelPlaceholder,
  "例如 decision-model-preview 或 jev-latest",
);
assert.equal(
  dictionaries.zh.routerCloudApiKeyEnv,
  "API Key 环境变量名（保护密钥）",
);
assert.equal(
  dictionaries.zh.routerCloudApiKeyEnvPlaceholder,
  "例如 DASHSCOPE_API_KEY（填入你在电脑里配的环境变量名）",
);
assert.equal(
  dictionaries.zh.pythonPathHint,
  "（选填/进阶向）一般留空即可！插件会自动帮你装配隔离环境。仅当自动装配失败，或你想强制指定某个 Python 时再填写。",
);
assert.equal(
  dictionaries.zh.hfHomeHint,
  "（选填）留空则自动存放到系统用户目录。如果你 C 盘爆满，可以在这里填个 D 盘或 E 盘的空文件夹路径。",
);

// --- Save payload / structure -------------------------------------------
for (const flatKey of [
  "router_enabled",
  "router_mode",
  "router_cloud_enabled",
  "router_cloud_provider",
  "router_cloud_base_url",
  "router_cloud_path",
  "router_cloud_model",
  "router_cloud_api_key_env",
]) {
  assert.ok(
    source.includes(`${flatKey}:`),
    `save payload is missing ${flatKey}`,
  );
  assert.ok(
    source.includes(`path: ['${flatKey}']`),
    `settings mutation is missing ${flatKey}`,
  );
}
assert.ok(source.includes("am-advanced"), "advanced collapsible block missing");
assert.ok(source.includes("am-advanced-notice"), "advanced notice missing");
assert.ok(
  source.includes("routerCloudEnabled ? 'custom' : ''"),
  "custom provider selection missing",
);

console.log("client_ui.test.mjs: 1 test passed");
