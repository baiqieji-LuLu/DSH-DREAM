/**
 * DSH Host half for the DSH-DREAM settings panel.
 *
 * The Host exports a `Config` schema with volatile fields that the Client
 * settings page reads and writes through `configForms`. On activation the
 * resolved values are also written to the Python bridge file next to the
 * installed plugin root.
 *
 * `rag_path` is an optional UI override. It is empty by default: the backend
 * resolves the per-user DSH path and never sends that default to the browser.
 * `AGENT_MEMORY_RAG_PATH` is the fallback when the field is blank.
 */
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import {
  applySessionEvent,
  createTurnState,
  finalizeTurn,
  isUsableDialogueText,
  messageText,
} from "./holographic_events.js";
import {
  PLUGIN_DIR,
  PROJECT_ROOT,
  RuntimeEnv,
  probeRuntime,
} from "./runtime_env.js";
import { PythonBridge } from "./bridge_process.js";
import { normalizePayload, redactPayload } from "./config_bridge.js";
import { registerMemoryTools } from "./memory_tools.js";
import { registerPreRetrieval } from "./pre_retrieval.js";
import { CloudRouter } from "./router_cloud.js";
import { createRouterEngine, resolveAuditPath } from "./router_engine.js";
import { LocalRouterClient } from "./router_local.js";

// DSH's plugin contract injects declared services into `ctx` before apply().
// Keep `tools` explicit so registration does not depend on a dynamic
// ctx.inject() callback firing correctly.
export const inject = ["tools"];

/** Profile entry id of the DSH settings form. */
const ENTRY_ID = "dsh-dream-settings";

/** Files used by the forced holographic turn bridge. */
const TURN_BRIDGE = resolve(PLUGIN_DIR, "turn_bridge.py");


/** Local activation log, useful when DSH's own runtime log is unavailable. */
const DIAG_PATH = resolve(
  process.env.DSH_DREAM_LOG || PROJECT_ROOT,
  "agent_memory_plugin.log",
);

function diag(message) {
  try {
    appendFileSync(
      DIAG_PATH,
      `${new Date().toISOString()} ${message}\n`,
      "utf8",
    );
  } catch {
    // Diagnostics must never be the reason the plugin fails to load.
  }
}

/**
 * Resolve Schemastery without assuming the linked package has its own
 * `node_modules`. A local `link:` bundle lives outside the profile tree, so a
 * bare ESM import can fail even though the profile has the package. The
 * profile-anchored fallback is what `dsh plugin add link:` installations use.
 */
const requireFromHere = createRequire(import.meta.url);
function loadSchema() {
  try {
    const schema = requireFromHere("@deepseek-ai/schemastery");
    diag("schemastery resolved from the linked package");
    return schema;
  } catch (firstError) {
    const profileDir =
      process.env.DSH_PROFILE_DIR ||
      resolve(
        process.env.USERPROFILE || process.env.HOME || ".",
        ".dsh",
        "profiles",
        "desktop",
      );
    try {
      const schema = createRequire(resolve(profileDir, "package.json"))(
        "@deepseek-ai/schemastery",
      );
      diag(`schemastery resolved from profile fallback: ${profileDir}`);
      return schema;
    } catch (secondError) {
      diag(
        `schemastery resolution failed; profileDir=${profileDir}; ` +
          `first=${firstError?.stack || firstError}; ` +
          `second=${secondError?.stack || secondError}`,
      );
      throw new Error(
        `DSH-DREAM cannot resolve @deepseek-ai/schemastery from ${import.meta.url}; ` +
          `profile fallback ${profileDir} also failed: ${secondError?.message || secondError}`,
        { cause: firstError },
      );
    }
  }
}
const z = loadSchema();

/** Absolute bridge file read by `agent_memory.config.load_config()`. */
export const CONFIG_BRIDGE_PATH = resolve(
  PROJECT_ROOT,
  "agent_memory_config.json",
);

/**
 * Target bridge path. Prefer the Host-specific override; deliberately ignore
 * the legacy global AGENT_MEMORY_CONFIG so a stale checkout cannot hijack the
 * current profile's config.
 */
const bridgePath = () =>
  resolve(process.env.DSH_DREAM_CONFIG || CONFIG_BRIDGE_PATH);

/** Read the existing bridge file, or return null when it is absent/invalid. */
function readBridgeFile(target) {
  try {
    if (!existsSync(target)) return null;
    const raw = JSON.parse(readFileSync(target, "utf8"));
    if (!raw || typeof raw !== "object") return null;
    return normalizePayload(raw);
  } catch (error) {
    diag(`bridge read failed: ${error?.stack || error}`);
    return null;
  }
}

export const Config = z.object({
  enable_memory: z
    .boolean()
    .default(true)
    .description(
      "Enable long-term memory. When off, memory_search returns empty without opening Chroma.",
    )
    .volatile(),

  enable_auto_recall: z
    .boolean()
    .default(true)
    .description(
      "Inject a small optional memory hint before the first model step only when a highly relevant memory exists. memory_search remains available when disabled.",
    )
    .volatile(),

  rag_path: z
    .string()
    .default("")
    .description(
      "Leave blank to use the DSH user directory; enter an absolute path to customize, or set AGENT_MEMORY_RAG_PATH to override.",
    )
    .volatile(),

  python_path: z
    .string()
    .default("")
    .description(
      "Optional path to a Python interpreter with chromadb and sentence-transformers; leave blank to auto-detect an adjacent .venv or PATH.",
    )
    .volatile(),

  hf_home_path: z
    .string()
    .default("")
    .description(
      "Optional writable model cache directory. Leave blank to try the DSH profile, plugin, and working-directory .hf-cache locations.",
    )
    .volatile(),

  max_window_tokens: z
    .natural()
    .min(128)
    .max(32768)
    .default(1500)
    .description("Token budget used to initialize WindowManager.")
    .volatile(),

  retrieval_top_k: z
    .natural()
    .min(1)
    .max(20)
    .default(5)
    .description("Number of RAG memories returned by memory_search.")
    .volatile(),

  embedding_provider: z
    .string()
    .default("local")
    .description(
      "Embedding provider: local (BAAI Chinese model) or openai_compatible (cloud /embeddings API).",
    )
    .volatile(),

  embedding_model: z
    .string()
    .default("")
    .description(
      "Cloud embedding model. Leave blank for local BAAI or cloud default text-embedding-3-small.",
    )
    .volatile(),

  embedding_api_key: z
    .string()
    .default("")
    .description(
      "Bearer token for the cloud embedding endpoint. Displayed masked; never logged.",
    )
    .volatile(),

  embedding_base_url: z
    .string()
    .default("")
    .description(
      "OpenAI-compatible embedding base URL; required for openai_compatible.",
    )
    .volatile(),

  auto_rebuild_collection: z
    .boolean()
    .default(true)
    .description(
      "Losslessly re-embed old records when the embedding dimension changes (cloud APIs may consume tokens).",
    )
    .volatile(),

  holographic_mode: z
    .boolean()
    .default(false)
    .description(
      "Force-write complete user and assistant originals for every turn, even when the agent calls no memory tool.",
    )
    .volatile(),

  // UI-only companion to `holographic_mode`: it shows a badge in the chat
  // area and can optionally tint that area, but it never changes what is
  // written. Disabling it must leave the interface exactly as it was before
  // this feature existed.
  holo_indicator: z
    .object({
      enabled: z
        .boolean()
        .default(true)
        .description(
          "Show the holographic recording indicator. This is a hint only, not the write switch.",
        ),
      style: z
        .union([z.const("badge"), z.const("theme")])
        .default("badge")
        .description(
          "badge shows only the small dot and label; theme also turns the chat container into white background with black text.",
        ),
    })
    .default({ enabled: true, style: "badge" })
    .description(
      "Visual hint for holographic mode. It does not control memory writes.",
    )
    .volatile(),

  // --- RouterEngine (v0.4.0) ---------------------------------------------
  // The RouterEngine only decides whether a turn needs long-term memory; it
  // never generates dialogue text. `router_enabled=false` restores the exact
  // pre-v0.4.0 behaviour, and `router_shadow_mode=true` records decisions
  // without ever intercepting retrieval.
  router_enabled: z
    .boolean()
    .default(true)
    .description(
      "Enable RouterEngine. When off, every non-empty turn retrieves memory exactly as before v0.4.0.",
    )
    .volatile(),

  router_mode: z
    .string()
    .default("auto")
    .description("Arbitration preference: auto, cloud, local, or regex.")
    .volatile(),

  router_budget_ms: z
    .natural()
    .min(50)
    .max(200)
    .default(200)
    .description(
      "Hard routing budget in milliseconds. Values above 200 are rejected; ambiguity and timeouts default to retrieve.",
    )
    .volatile(),

  router_threshold: z
    .number()
    .min(0)
    .max(1)
    .default(0.55)
    .description("Probability threshold used by the cloud and regex layers.")
    .volatile(),

  router_uncertainty_band: z
    .number()
    .min(0)
    .max(0.3)
    .default(0.1)
    .description(
      "Abstention band around the threshold. Inside the band the engine retrieves (better to over-retrieve than to forget).",
    )
    .volatile(),

  router_timeout_policy: z
    .string()
    .default("retrieve")
    .description(
      "Decision after the 200 ms deadline: retrieve (default, neutral) or bypass_main.",
    )
    .volatile(),

  router_shadow_mode: z
    .boolean()
    .default(false)
    .description(
      "Shadow mode: always retrieve like the legacy path, but log what RouterEngine would have decided.",
    )
    .volatile(),

  router_cloud_enabled: z
    .boolean()
    .default(false)
    .description(
      "Enable the cloud decision model. Off by default: the query text would leave this machine.",
    )
    .volatile(),

  router_cloud_provider: z
    .string()
    .default("")
    .description(
      "Active cloud provider preset: typesafe, ali-bailian, tokenrhythm, openai, local-laya, or custom.",
    )
    .volatile(),

  router_cloud_base_url: z
    .string()
    .default("")
    .description("Cloud decision endpoint base URL (for example a workspace-level endpoint).")
    .volatile(),

  router_cloud_path: z
    .string()
    .default("")
    .description("Decision endpoint path, for example /v1/systemone.")
    .volatile(),

  router_cloud_model: z
    .string()
    .default("")
    .description("Decision model name for the active provider.")
    .volatile(),

  router_cloud_api_key_env: z
    .string()
    .default("")
    .description(
      "Environment variable holding the provider credential. The key itself is never stored here.",
    )
    .volatile(),

  router_local_backend: z
    .string()
    .default("bge-head")
    .description("Local router backend: bge-head (default), onnx-intent, or off.")
    .volatile(),
});

/** Read the authoritative live values and revision from the settings service. */
function settingsDescriptor(settings, previous = null) {
  if (!settings || typeof settings.describe !== "function") return null;
  try {
    const forms = settings.describe();
    if (!Array.isArray(forms)) return null;
    const entry = forms.find((form) => {
      const ns = String(form?.ns || "");
      return (
        ns === ENTRY_ID ||
        ns.endsWith(`:${ENTRY_ID}`) ||
        ns.endsWith(`/${ENTRY_ID}`)
      );
    });
    if (!entry || !entry.value || typeof entry.value !== "object") return null;
    return {
      // The form may not render every router setting; `previous` keeps the
      // advanced values (budget, threshold, cloud endpoint, circuit, audit).
      payload: normalizePayload(entry.value, previous),
      revision: Number(entry.revision) || 0,
    };
  } catch (error) {
    diag(`settings.describe failed: ${error?.stack || error}`);
    return null;
  }
}

export function apply(ctx, config) {
  try {
    diag(`apply start; keys=${Object.keys(config || {}).join(",")}`);

    const target = bridgePath();
    const existing = readBridgeFile(target);
    let allowSync = existing === null;
    let baselineRevision = null;
    let lastJson = existing ? JSON.stringify(existing) : "";

    const writeBridge = (payload) => {
      try {
        // The browser form posts only the fields it renders; keep the router
        // block from the current file when the payload did not carry one.
        const previousRouter = (() => {
          try {
            return readBridgeFile(target)?.router ?? null;
          } catch {
            return null;
          }
        })();
        const merged = {
          ...payload,
          router:
            payload?.router ?? previousRouter ?? normalizePayload({}).router,
        };
        const json = `${JSON.stringify(merged, null, 2)}\n`;
        if (json === lastJson) return;
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, json, "utf8");
        lastJson = json;
        diag(
          `bridge written: ${target}; payload=${JSON.stringify(
            redactPayload(merged),
          )}`,
        );
        ctx.logger?.info?.("DSH-DREAM config bridge updated: %s", target);
      } catch (error) {
        // Keep settings UI available even when the bridge path is not writable.
        diag(`bridge write failed: ${error?.stack || error}`);
        ctx.logger?.error?.(
          "DSH-DREAM config bridge could not write %s: %s",
          target,
          error,
        );
      }
    };

    // Holographic mode: observe committed turn messages and force a write at
    // turn end even when the Agent never calls a memory tool.
    const holoLog = (message) => {
      const line = `[Holo] ${message}`;
      diag(line);
      console.log(line);
    };
    const preview = (value) => {
      const text = String(value || "").replace(/\s+/g, " ").trim();
      return text.length > 180 ? `${text.slice(0, 180)}…` : text;
    };
    const turnStates = new Map();
    const runtime = new RuntimeEnv({ target, readBridgeFile, diag });
    const bridge = new PythonBridge({
      target,
      runtime,
      getConfig: () => readBridgeFile(target) || {},
      bridgePath: TURN_BRIDGE,
      projectRoot: PROJECT_ROOT,
      diag,
      onLog: holoLog,
    });
    const queryMemory = (query, topK, timeoutMs, signal, timelineMode) =>
      bridge.query(query, topK, timeoutMs, signal, timelineMode);
    const writeTurn = (user, assistant, extractedFact = null) =>
      bridge.writeTurn(user, assistant, extractedFact);
    const sendHolographicTurn = (user, assistant) => {
      holoLog(
        `准备发往 Python 桥接 user=${preview(user)} assistant=${preview(assistant)}`,
      );
      bridge
        .writeTurn(user, assistant)
        .then((ok) => {
          if (ok) holoLog("已写入 Python 桥接 stdin");
          else holoLog("Python 桥接不可用，本轮未写入");
        })
        .catch((error) => holoLog(`发往 Python 桥接失败: ${error?.stack || error}`));
      return true;
    };

    // --- RouterEngine (v0.4.0) -----------------------------------------
    // The router only says "retrieve" or "skip"; it never writes dialogue or
    // prompt text. Its audit log stores a query hash by default and lives at
    // the documented path (`~/.dsh/dream/router.jsonl`, overridable with
    // DSH_DREAM_ROUTER_AUDIT) so operators can tail one known file.
    const auditPath = resolveAuditPath();
    const resolveRouterCredential = async (envName) => {
      if (!envName) return "";
      try {
        const credentials =
          typeof ctx.get === "function" ? ctx.get("credentials") : undefined;
        if (credentials && typeof credentials.resolve === "function") {
          const resolved = await credentials.resolve(envName);
          const text =
            typeof resolved === "string"
              ? resolved
              : resolved?.value ?? resolved?.secret ?? "";
          if (text) return String(text);
        }
      } catch (error) {
        diag(`[Router] credential service failed: ${error?.stack || error}`);
      }
      return String(process.env[envName] || "");
    };
    const localRouter = new LocalRouterClient({
      bridge,
      getConfig: () => readBridgeFile(target) || {},
      diag,
    });
    const cloudRouter = new CloudRouter({
      getConfig: () => readBridgeFile(target) || {},
      resolveCredential: resolveRouterCredential,
      diag,
    });
    const router = createRouterEngine({
      getConfig: () => readBridgeFile(target) || {},
      local: localRouter,
      cloud: cloudRouter,
      auditPath,
      diag,
    });
    diag(`[Router] audit log path: ${auditPath}`);
    console.log(`[Router] audit log path: ${auditPath}`);
    // Create the file immediately (independent of the Python runtime) so an
    // operator can start `Get-Content -Wait` before the first turn.
    void router.ensureAuditLog();

    registerMemoryTools(ctx, { queryMemory, writeTurn, diag });
    registerPreRetrieval(ctx, {
      queryMemory,
      holoLog,
      diag,
      getConfig: () => readBridgeFile(target) || {},
      router,
    });

    runtime
      .ensureReady()
      .then(async (state) => {
        if (!["ready", "installed"].includes(state.status)) return null;
        const child = await bridge.ensure();
        if (child) {
          // Warmup is the only place that may load the local encoder; it runs
          // after the bridge is ready and never from agent/pre-step.
          await router.warmup();
          void localRouter.status().catch(() => null);
        }
        return child;
      })
      .catch((error) => diag(`runtime warmup failed: ${error?.stack || error}`));

    const onSessionEvent = (session, event) => {
      if (!session || !event || !event.type) return;
      const id = String(session.id);
      let state = turnStates.get(id);
      if (!state) {
        state = createTurnState();
        turnStates.set(id, state);
      }
      if (event.type === "user/message") {
        const text = messageText(event.data);
        const role = event.data?.role || "user";
        if (!isUsableDialogueText(role, text)) {
          holoLog(`被过滤逻辑拦截: event=user/message role=${role} text=${preview(text)}`);
        } else {
          holoLog(`捕获 user_text=${preview(text)}`);
        }
      }
      if (event.type === "assistant/message") {
        const text = messageText(event.data?.message);
        const role = event.data?.message?.role || "assistant";
        if (!isUsableDialogueText(role, text)) {
          holoLog(`被过滤逻辑拦截: event=assistant/message role=${role} text=${preview(text)}`);
        } else {
          holoLog(`捕获 assistant_text=${preview(text)}`);
        }
      }
      applySessionEvent(state, event);
    };

    const flushHolographicTurn = (id, turnNumber, attempt = 0) => {
      const state = turnStates.get(id);
      if (!state || state.flushed) return;
      holoLog(
        `flush attempt=${attempt} turn=${turnNumber} buffered_user=${preview(state.user)} buffered_assistant=${preview(state.assistant)}`,
      );
      const configNow = readBridgeFile(target);
      if (!configNow?.holographic_mode) {
        holoLog("被过滤逻辑拦截: holographic_mode=false");
        turnStates.delete(id);
        return;
      }
      if (state.manualSucceeded) {
        holoLog("跳过自动写入: Agent 已成功调用 memory_add_turn");
        turnStates.delete(id);
        return;
      }

      // turn-stopping can arrive before our incremental buffer has the final
      // assistant text. Recover it from committed events and retry briefly
      // while the turn boundary commits.
      const sessions =
        typeof ctx.get === "function" ? ctx.get("sessions") : undefined;
      const liveSession =
        sessions && typeof sessions.get === "function"
          ? sessions.get(id)
          : undefined;
      const events =
        liveSession && typeof liveSession.snapshotEvents === "function"
          ? liveSession.snapshotEvents()
          : [];
      const turn = finalizeTurn(state, events, turnNumber);
      if (!turn) {
        if (attempt < 20) {
          setTimeout(
            () => flushHolographicTurn(id, turnNumber, attempt + 1),
            50,
          );
        } else {
          turnStates.delete(id);
          holoLog("被过滤逻辑拦截: 缺少干净的 user/assistant 文本");
        }
        return;
      }
      holoLog(`finalized user=${preview(turn.user)} assistant=${preview(turn.assistant)}`);
      state.flushed = true;
      turnStates.delete(id);
      const sent = sendHolographicTurn(turn.user, turn.assistant);
      if (!sent) diag("holographic turn was not sent");
    };

    const onTurnStopping = (payload) => {
      const id = String(payload?.agent?.id || "");
      if (!id) return;
      holoLog(`捕获到 turn-stopping 事件 agent=${id} turn=${payload?.turn}`);
      const state = turnStates.get(id);
      if (!state) {
        holoLog("被过滤逻辑拦截: 没有该 session 的 turn 缓存");
        return;
      }
      flushHolographicTurn(id, payload.turn);
    };

    ctx.effect(
      () => () => {
        router.close();
        bridge.close();
        turnStates.clear();
      },
      "DSH-DREAM.holographic-bridge",
    );

    // File-first startup: if the bridge exists, preserve it. Only a missing
    // bridge is initialized from the current Config/default values.
    if (existing === null) {
      diag("bridge missing; initializing from Config defaults");
      writeBridge(normalizePayload(config));
    } else {
      diag(
        `bridge exists; startup preserves payload=${JSON.stringify(
          redactPayload(existing),
        )}`,
      );
    }

    const syncFromSettings = (settings, eventRevision) => {
      const descriptor = settingsDescriptor(settings, readBridgeFile(target));
      if (!descriptor) return false;
      if (
        eventRevision !== undefined &&
        baselineRevision !== null &&
        Number(eventRevision) <= baselineRevision
      ) {
        diag(
          `ignore stale settings event rev=${eventRevision} baseline=${baselineRevision}`,
        );
        return false;
      }
      baselineRevision = descriptor.revision;
      allowSync = true;
      writeBridge(descriptor.payload);
      return true;
    };

    const timer = setInterval(() => {
      if (!allowSync) return;
      const settings =
        typeof ctx.get === "function" ? ctx.get("settings") : undefined;
      if (settings) syncFromSettings(settings);
    }, 1000);
    timer.unref?.();

    ctx.effect(
      () => () => clearInterval(timer),
      "DSH-DREAM.config-bridge-timer",
    );

    // Hard bridge: the browser posts the exact values it just saved. This path
    // does not depend on settings events or revisions and writes the JSON
    // immediately. It is same-origin and only registered by this plugin.
    ctx.inject(["sessions"], (child) => {
      child.effect(
        () => {
          const offSession = child.on("session/event", onSessionEvent);
          const offStop = child.on("agent/turn-stopping", onTurnStopping);
          return () => {
            offSession?.();
            offStop?.();
          };
        },
        "DSH-DREAM.holographic-events",
      );
    });

    ctx.inject(["webServer"], (child) => {
      child.effect(
        () => {
          const disposeRuntime = child.webServer.register({
            kind: "exact",
            path: "/dsh-dream/runtime",
            handler: async (req, res) => {
              if (req.method !== "GET") {
                res.writeHead(405, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: "GET required" }));
                return;
              }
              try {
                const bridgeConfig = readBridgeFile(target);
                const status = await probeRuntime(
                  bridgeConfig?.python_path,
                  bridgeConfig?.hf_home_path,
                );
                res.writeHead(200, { "content-type": "application/json" });
                res.end(
                  JSON.stringify({
                    ...status,
                    provision: runtime.state,
                  }),
                );
              } catch (error) {
                res.writeHead(500, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: String(error) }));
              }
            },
          });
          const dispose = child.webServer.register({
            kind: "exact",
            path: "/dsh-dream/config",
            handler: async (req, res) => {
              if (req.method !== "POST") {
                res.writeHead(405, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: "POST required" }));
                return;
              }
              try {
                let raw = "";
                for await (const chunk of req) raw += chunk;
                const parsed = JSON.parse(raw || "{}");
                const previous = readBridgeFile(target);
                const payload = normalizePayload(parsed, previous);
                console.log(
                  "🔥 [Host] HTTP 收到配置",
                  redactPayload(payload),
                );
                writeBridge(payload);
                res.writeHead(200, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: true, payload }));
              } catch (error) {
                console.error("🔥 [Host] HTTP 配置写入失败", error);
                diag(`http bridge failed: ${error?.stack || error}`);
                res.writeHead(400, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: String(error) }));
              }
            },
          });
          const disposeRouter = child.webServer.register({
            kind: "exact",
            path: "/dsh-dream/router",
            handler: async (req, res) => {
              if (req.method !== "GET") {
                res.writeHead(405, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: "GET required" }));
                return;
              }
              try {
                const status = await router.status();
                status.auditPath = auditPath;
                res.writeHead(200, { "content-type": "application/json" });
                res.end(JSON.stringify(status));
              } catch (error) {
                res.writeHead(500, { "content-type": "application/json" });
                res.end(JSON.stringify({ ok: false, error: String(error) }));
              }
            },
          });
          return () => {
            dispose?.();
            disposeRuntime?.();
            disposeRouter?.();
          };
        },
        "DSH-DREAM.http-bridge",
      );
    });

    ctx.inject(["settings"], (child) => {
      child.effect(
        () => {
          const initial = settingsDescriptor(
            child.settings,
            readBridgeFile(target),
          );
          if (initial) baselineRevision = initial.revision;
          diag(
            `settings injected; baselineRevision=${baselineRevision}; allowSync=${allowSync}`,
          );

          const offSettings = child.on(
            "settings/document-updated",
            (ns, revision) => {
              if (
                String(ns) === ENTRY_ID ||
                String(ns).endsWith(`:${ENTRY_ID}`) ||
                String(ns).endsWith(`/${ENTRY_ID}`)
              ) {
                const descriptor = settingsDescriptor(
                  child.settings,
                  readBridgeFile(target),
                );
                console.log(
                  "🔥 [Host] 收到设置更新事件",
                  ns,
                  revision,
                  descriptor ? redactPayload(descriptor.payload) : descriptor,
                );
                diag(`settings event ns=${ns} revision=${revision}`);
                syncFromSettings(child.settings, revision);
              }
            },
          );
          const offReload = child.on("app-boot/config-reload", () => {
            if (allowSync) syncFromSettings(child.settings);
          });
          return () => {
            offSettings?.();
            offReload?.();
          };
        },
        "DSH-DREAM.config-bridge-settings",
      );
    });
  } catch (error) {
    diag(`apply failed: ${error?.stack || error}`);
    throw error;
  }
}
