/**
 * Holographic-indicator behaviour tests (no DSH runtime, no browser).
 *
 * The component is captured from the real `client.js` registration and rendered
 * through a deliberately small React stand-in. The tests pin the states that
 * matter for the acceptance criteria:
 *
 * - holographic mode off            -> no DOM at all;
 * - indicator disabled             -> no DOM at all;
 * - badge enabled                  -> dot + label + aria, no theme surface;
 * - legacy config                  -> badge default.
 *
 * Drag wiring is pinned statically: the badge must be an absolute overlay with
 * pointer capture and a persisted frame-relative position.
 *
 * Run with:  node tests/holo_indicator.test.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const clientPath = fileURLToPath(
  new URL("../dsh_plugin/client.js", import.meta.url),
);
const source = readFileSync(clientPath, "utf8");

// --- A minimal React stand-in --------------------------------------------
let snapshot = { status: "ready", value: {} };
let created = [];

const fakeReact = {
  Fragment: Symbol("Fragment"),
  createElement(type, props, ...children) {
    const element = { type, props: props || {}, children };
    created.push(element);
    return element;
  },
  useState: (initial) => [initial, () => {}],
  useEffect: () => {},
  useLayoutEffect: () => {},
  useCallback: (fn) => fn,
  useRef: (initial) => ({ current: initial }),
  useSyncExternalStore: () => snapshot,
};

let moduleEntry = null;
globalThis.window = {
  __ModuleLoader__: {
    load: (entry) => {
      moduleEntry = entry;
    },
  },
};

new Function("window", "React", source)(globalThis.window, fakeReact);
const exported = moduleEntry.factory((name) => {
  if (name === "react") return fakeReact;
  throw new Error(`unexpected require(${name})`);
});

const dictionaries = {};
const components = {};
const scope = {
  getSnapshot: () => snapshot,
  subscribe: () => () => {},
};
const ctx = {
  effect: (fn) => fn(),
  locale: {
    bind: () => (key) => key,
    register: (ns, value) => {
      dictionaries[ns] = value;
      return () => {};
    },
  },
  configForms: {
    get: () => scope,
    whileServed: (_ids, callback) => callback(),
  },
  slots: {
    inject: (_name, callback) => callback(),
    register: (meta, component) => {
      components[meta.name] = component;
      return () => {};
    },
  },
};
exported.apply(ctx);

const HoloIndicator = components["shell.overlay"];
assert.equal(
  typeof HoloIndicator,
  "function",
  "shell.overlay indicator was not registered",
);

const t = (key) => dictionaries["settings.agentMemory"].zh[key] ?? key;

function render(value) {
  snapshot = { status: "ready", value };
  created = [];
  const result = HoloIndicator({ t, scope });
  return { result, elements: created };
}

function badgeOf(elements) {
  return elements.find(
    (element) => element.props?.className === "dsh-dream-holo-indicator",
  );
}

// --- Acceptance-shaped states -------------------------------------------
let run = render({
  holographic_mode: false,
  holo_indicator: { enabled: true },
});
assert.equal(run.result, null, "holographic off must render nothing");
assert.equal(run.elements.length, 0, "holographic off must create no elements");

run = render({
  holographic_mode: true,
  holo_indicator: { enabled: false },
});
assert.equal(run.result, null, "indicator disabled must render nothing");
assert.equal(run.elements.length, 0, "indicator disabled must create no elements");

run = render({ holographic_mode: true });
assert.ok(run.result, "legacy config must fall back to the badge");
assert.ok(run.elements.some((element) => element.type === "style"));
const badge = badgeOf(run.elements);
assert.ok(badge, "badge element is missing");
assert.equal(badge.props.role, "status");
assert.equal(badge.props.tabIndex, 0);
assert.equal(badge.props["aria-keyshortcuts"], "ArrowLeft ArrowRight ArrowUp ArrowDown");
assert.match(badge.props["aria-label"], /拖动可调整位置/);
assert.equal(badge.props["aria-label"].startsWith("全息模式已开启"), true);
assert.equal(badge.props.style.visibility, "hidden");
const badgeText = badge.children.find(
  (child) => child?.props?.className === "dsh-dream-holo-indicator-text",
);
assert.equal(badgeText?.children?.[0], "全息 · 落库中");
assert.equal(
  badge.children.some(
    (child) => child?.props?.className === "dsh-dream-holo-indicator-dot",
  ),
  true,
  "badge dot is missing",
);

// --- The removed theme variant must stay removed -------------------------
assert.equal(source.includes("dsh-dream-holo-theme"), false);
assert.equal(source.includes("holoStyleTheme"), false);

// --- Draggable frame overlay wiring --------------------------------------
assert.ok(source.includes('[data-conversation-region="chat"]'));
assert.ok(source.includes("[data-shell-overlay]"));
assert.ok(source.includes("setPointerCapture"));
assert.ok(source.includes("localStorage"));
assert.ok(source.includes("cursor: grab"));
assert.ok(source.includes("position: absolute"));
assert.ok(source.includes("onPointerDown"));
assert.ok(source.includes("onPointerMove"));
assert.ok(source.includes("onPointerUp"));

// --- The badge CSS stays scoped -----------------------------------------
const cssStart = source.indexOf("const HOLO_CSS = `");
const cssEnd = source.indexOf("`;", cssStart);
const holoCss = source.slice(cssStart, cssEnd);
assert.ok(holoCss.includes(".dsh-dream-holo-indicator"));
assert.ok(holoCss.includes("cursor: grab"));
assert.ok(holoCss.includes("pointer-events: auto"));
assert.ok(
  !/(^|\n)\s*(body|html|:root)\b/.test(holoCss),
  "badge CSS must not target body/html/:root",
);

console.log("holo_indicator.test.mjs: 1 test passed");
