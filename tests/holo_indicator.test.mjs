/**
 * Holographic-indicator behaviour tests (no DSH runtime, no browser).
 *
 * The component is captured from the real `client.js` registration and rendered
 * through a deliberately small React stand-in. The tests pin the four states
 * that matter for the acceptance criteria:
 *
 * - holographic mode off            -> no DOM at all;
 * - indicator disabled             -> no DOM at all;
 * - badge style                    -> dot + label + aria, no theme class;
 * - theme style                    -> chat container class, clean removal.
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

// --- A minimal React + DOM stand-in --------------------------------------
let snapshot = { status: "ready", value: {} };
let created = [];
let layoutEffects = [];
const themeClasses = new Set();

const fakeChatHost = {
  classList: {
    add: (name) => themeClasses.add(name),
    remove: (name) => themeClasses.delete(name),
    contains: (name) => themeClasses.has(name),
  },
};

const fakeAnchor = {
  closest: (selector) =>
    selector === '[data-conversation-region="chat"]' ? fakeChatHost : null,
};

const fakeReact = {
  Fragment: Symbol("Fragment"),
  createElement(type, props, ...children) {
    const element = { type, props: props || {}, children };
    created.push(element);
    if (props?.ref && typeof props.ref === "object") {
      props.ref.current = fakeAnchor;
    }
    return element;
  },
  useState: (initial) => [initial, () => {}],
  useEffect: () => {},
  useLayoutEffect: (fn, deps) => {
    layoutEffects.push({ fn, deps });
  },
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

const HoloIndicator = components["conversation.input.dock"];
assert.equal(
  typeof HoloIndicator,
  "function",
  "conversation.input.dock indicator was not registered",
);

const t = (key) => dictionaries["settings.agentMemory"].zh[key] ?? key;

function render(value) {
  snapshot = { status: "ready", value };
  created = [];
  layoutEffects = [];
  themeClasses.clear();
  const result = HoloIndicator({ t, scope, sessionId: "session-test" });
  return { result, elements: created, effects: layoutEffects };
}

function badgeOf(elements) {
  return elements.find(
    (element) =>
      element.props?.className === "dsh-dream-holo-indicator",
  );
}

// --- Acceptance-shaped states -------------------------------------------
let run = render({
  holographic_mode: false,
  holo_indicator: { enabled: true, style: "badge" },
});
assert.equal(run.result, null, "holographic off must render nothing");
assert.equal(run.elements.length, 0, "holographic off must create no elements");

run = render({
  holographic_mode: true,
  holo_indicator: { enabled: false, style: "theme" },
});
assert.equal(run.result, null, "indicator disabled must render nothing");
assert.equal(run.elements.length, 0, "indicator disabled must create no elements");
assert.equal(themeClasses.size, 0, "indicator disabled must not theme the chat");

run = render({ holographic_mode: true });
assert.ok(run.result, "legacy config must fall back to the badge");
assert.ok(run.elements.some((element) => element.type === "style"));
const badge = badgeOf(run.elements);
assert.ok(badge, "badge element is missing");
assert.equal(
  badge.props["aria-label"],
  dictionaries["settings.agentMemory"].zh.holoIndicatorAria,
);
const badgeText = badge.children.find(
  (child) => child?.props?.className === "dsh-dream-holo-indicator-text",
);
assert.equal(
  badgeText?.children?.[0],
  dictionaries["settings.agentMemory"].zh.holoIndicatorOn,
);
assert.equal(
  badge.children.some(
    (child) => child?.props?.className === "dsh-dream-holo-indicator-dot",
  ),
  true,
  "badge dot is missing",
);
run.effects.forEach((effect) => effect.fn());
assert.equal(themeClasses.size, 0, "badge style must not theme the chat");

run = render({
  holographic_mode: true,
  holo_indicator: { enabled: true, style: "theme" },
});
assert.ok(run.result, "theme style must render the badge");
const cleanup = run.effects[0].fn();
assert.equal(
  themeClasses.has("dsh-dream-holo-theme"),
  true,
  "theme class was not applied to the chat container",
);
cleanup();
assert.equal(
  themeClasses.has("dsh-dream-holo-theme"),
  false,
  "theme class was not removed on cleanup",
);
assert.equal(themeClasses.size, 0, "theme cleanup left variables behind");

// --- Skin safety: the theme CSS stays scoped ----------------------------
const cssStart = source.indexOf("const HOLO_CSS = `");
const cssEnd = source.indexOf("`;", cssStart);
const holoCss = source.slice(cssStart, cssEnd);
assert.ok(holoCss.includes(".dsh-dream-holo-theme"));
assert.ok(
  !/(^|\n)\s*(body|html|:root)\b/.test(holoCss),
  "theme CSS must not target body/html/:root",
);

console.log("holo_indicator.test.mjs: 1 test passed");
