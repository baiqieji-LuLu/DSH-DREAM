/**
 * DSH Host half for the DSH-DREAM settings panel.
 *
 * The Host exports a `Config` schema with volatile fields that the Client
 * settings page reads and writes through `configForms`. On activation the
 * resolved values are also written to the absolute Python bridge file
 * `G:\deepseek_ragproject_workspace\agent_memory_config.json`.
 *
 * `rag_path` is disabled (read-only) in the panel and defaults to the Python
 * project's Chroma directory. Set `AGENT_MEMORY_RAG_PATH` before starting DSH
 * to point the panel at another directory.
 */
import { spawn } from "node:child_process";
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  applySessionEvent,
  createTurnState,
  finalizeTurn,
  isUsableDialogueText,
  messageText,
} from "./holographic_events.js";

/** Absolute Python project root shared by the DSH panel and the Python package. */
const PROJECT_ROOT = "G:\\deepseek_ragproject_workspace";

/** Profile entry id of the DSH settings form. */
const ENTRY_ID = "dsh-dream-settings";

/** Files used by the forced holographic turn bridge. */
const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url));
const TURN_BRIDGE = resolve(PLUGIN_DIR, "turn_bridge.py");
const PYTHON = resolve(PROJECT_ROOT, ".venv", "Scripts", "python.exe");

/** Local activation log, useful when DSH's own runtime log is unavailable. */
const DIAG_PATH = resolve(PROJECT_ROOT, "agent_memory_plugin.log");

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

const DEFAULT_RAG_PATH = resolve(
  process.env.AGENT_MEMORY_RAG_PATH || resolve(PROJECT_ROOT, ".chroma"),
);

/**
 * Target bridge path. The default is the fixed absolute path above; an
 * explicit `AGENT_MEMORY_CONFIG` may override it for a non-default checkout.
 */
const bridgePath = () =>
  resolve(process.env.AGENT_MEMORY_CONFIG || CONFIG_BRIDGE_PATH);

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

  rag_path: z
    .string()
    .default(DEFAULT_RAG_PATH)
    .description(
      "Absolute Chroma persistence path. Read-only; set AGENT_MEMORY_RAG_PATH before starting DSH to change it.",
    )
    .disabled()
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
});

/**
 * Read a Loader config field that may be a plain value or a volatile reference.
 * DSH delivers either form depending on whether the settings service is active,
 * so calling `.get()` unconditionally is not safe.
 */
const current = (value) =>
  value !== null && typeof value === "object" && typeof value.get === "function"
    ? value.get()
    : value;

/** Normalize a raw Config or Settings descriptor value for the Python bridge. */
function normalizePayload(raw) {
  const value = raw && typeof raw === "object" ? raw : {};
  const provider = String(current(value.embedding_provider) ?? "local")
    .trim()
    .toLowerCase();
  return {
    enable_memory: Boolean(current(value.enable_memory) ?? true),
    rag_path: resolve(current(value.rag_path) || DEFAULT_RAG_PATH),
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
  };
}

/** Never write a bearer token into logs. */
function redactPayload(payload) {
  return {
    ...payload,
    embedding_api_key: payload.embedding_api_key ? "***" : "",
  };
}

/** Read the authoritative live values and revision from the settings service. */
function settingsDescriptor(settings) {
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
      payload: normalizePayload(entry.value),
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
        const json = `${JSON.stringify(payload, null, 2)}\n`;
        if (json === lastJson) return;
        mkdirSync(dirname(target), { recursive: true });
        writeFileSync(target, json, "utf8");
        lastJson = json;
        diag(
          `bridge written: ${target}; payload=${JSON.stringify(
            redactPayload(payload),
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
    let turnBridge = null;
    let turnBridgeBuffer = "";

    const ensureTurnBridge = () => {
      if (turnBridge && !turnBridge.killed) return turnBridge;
      try {
        turnBridge = spawn(PYTHON, ["-u", TURN_BRIDGE], {
          cwd: PROJECT_ROOT,
          env: { ...process.env, AGENT_MEMORY_CONFIG: target },
          stdio: ["pipe", "pipe", "pipe"],
          windowsHide: true,
        });
        turnBridge.stdout.setEncoding("utf8");
        turnBridge.stdout.on("data", (chunk) => {
          turnBridgeBuffer += chunk;
          const lines = turnBridgeBuffer.split(/\r?\n/);
          turnBridgeBuffer = lines.pop() || "";
          for (const line of lines) {
            if (line.trim()) diag(`holographic bridge: ${line.trim()}`);
          }
        });
        turnBridge.stderr.setEncoding("utf8");
        turnBridge.stderr.on("data", (chunk) => {
          diag(`holographic bridge stderr: ${String(chunk).trim()}`);
        });
        turnBridge.on("error", (error) => {
          diag(`holographic bridge spawn failed: ${error?.stack || error}`);
          turnBridge = null;
        });
        turnBridge.on("exit", (code, signal) => {
          diag(`holographic bridge exited code=${code} signal=${signal}`);
          turnBridge = null;
        });
      } catch (error) {
        diag(`holographic bridge start failed: ${error?.stack || error}`);
        turnBridge = null;
      }
      return turnBridge;
    };

    const sendHolographicTurn = (user, assistant) => {
      holoLog(`准备发往 Python 桥接 user=${preview(user)} assistant=${preview(assistant)}`);
      const bridge = ensureTurnBridge();
      if (!bridge || !bridge.stdin || bridge.stdin.destroyed) {
        holoLog("Python 桥接不可用，本轮未写入");
        return false;
      }
      try {
        bridge.stdin.write(
          `${JSON.stringify({ user_msg: user, ai_msg: assistant })}\n`,
        );
        holoLog("已写入 Python 桥接 stdin");
        return true;
      } catch (error) {
        holoLog(`发往 Python 桥接失败: ${error?.stack || error}`);
        return false;
      }
    };

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
        try {
          turnBridge?.kill();
        } catch {
          // Child cleanup must not fail plugin disposal.
        }
        turnBridge = null;
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
      const descriptor = settingsDescriptor(settings);
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
                const payload = normalizePayload(parsed);
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
          return () => dispose?.();
        },
        "DSH-DREAM.http-bridge",
      );
    });

    ctx.inject(["settings"], (child) => {
      child.effect(
        () => {
          const initial = settingsDescriptor(child.settings);
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
                const descriptor = settingsDescriptor(child.settings);
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
