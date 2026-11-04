/**
 * Persistent line-oriented Python bridge: spawn, warmup, query, write.
 * The environment gate lives in runtime_env.js; this module never provisions.
 */
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { resolveCache, resolvePython } from "./runtime_env.js";

export class PythonBridge {
  constructor({
    target,
    runtime,
    getConfig,
    bridgePath,
    projectRoot,
    diag = () => {},
    onLog = () => {},
  }) {
    this.target = target;
    this.runtime = runtime;
    this.getConfig = getConfig;
    this.bridgePath = bridgePath;
    this.projectRoot = projectRoot;
    this.diag = diag;
    this.onLog = onLog;
    this.child = null;
    this.buffer = "";
    this.pending = new Map();
    this.coldStart = true;
    this.startPromise = null;
    this.closed = false;
  }

  _log(level, message) {
    const line = `[Bridge] ${message}`;
    if (level === "error") console.error(line);
    else console.log(line);
    this.diag(line);
  }

  async ensure() {
    if (this.closed) return null;
    if (this.child && !this.child.killed) return this.child;
    if (this.startPromise) return this.startPromise;
    this.startPromise = (async () => {
      const envState = await this.runtime.ensureReady();
      if (!["ready", "installed"].includes(envState.status)) {
        this._log(
          "error",
          `runtime not ready: status=${envState.status} error=${
            envState.error || "unknown"
          }`,
        );
        return null;
      }
      const config = this.getConfig() || {};
      const cache = resolveCache(config.hf_home_path);
      if (!cache.path) {
        this._log("error", `cache unavailable: ${cache.error || "unknown"}`);
        return null;
      }
      const python = envState.python || resolvePython(config.python_path);
      this._log("info", `Spawning: ${python} ${this.bridgePath}`);
      let child;
      try {
        child = spawn(python, ["-u", this.bridgePath], {
        cwd: this.projectRoot,
        env: {
          ...process.env,
          AGENT_MEMORY_CONFIG: this.target,
          DSH_DREAM_CONFIG: this.target,
          DSH_DREAM_PYTHON: python,
          DSH_DREAM_RESOLVED_PYTHON: python,
          DSH_DREAM_HOST_PROVISION: "1",
          DSH_DREAM_HF_CACHE: cache.path,
          HF_HOME: cache.path,
          HF_HUB_CACHE: resolve(cache.path, "hub"),
          HF_HUB_OFFLINE: "0",
          TRANSFORMERS_OFFLINE: "0",
        },
          stdio: ["pipe", "pipe", "pipe"],
          windowsHide: true,
        });
      } catch (error) {
        this._log("error", `spawn threw: ${error?.stack || error}`);
        return null;
      }
      this.child = child;
      this.coldStart = true;
      this.buffer = "";
      child.stdout.setEncoding("utf8");
      child.stdout.on("data", (chunk) => {
        this.buffer += chunk;
        const lines = this.buffer.split(/\r?\n/);
        this.buffer = lines.pop() || "";
        for (const line of lines) this._onLine(line);
      });
      child.stderr.setEncoding("utf8");
      child.stderr.on("data", (chunk) => {
        const text = String(chunk).trim();
        this._log("error", `Stderr: ${text}`);
      });
      child.on("error", (error) => {
        this._log("error", `spawn failed: ${error?.stack || error}`);
        this.child = null;
      });
      child.on("exit", (code, signal) => {
        this._log("info", `Exited with code=${code} signal=${signal}`);
        this.child = null;
      });
      try {
        child.stdin.write(
          `${JSON.stringify({ op: "warmup", request_id: "warmup" })}\n`,
        );
      } catch (error) {
        this._log("error", `warmup write failed: ${error?.stack || error}`);
      }
      return child;
    })().finally(() => {
      this.startPromise = null;
    });
    return this.startPromise;
  }

  _onLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return;
    try {
      const response = JSON.parse(trimmed);
      if (response?.request_id) {
        const pending = this.pending.get(response.request_id);
        if (pending) {
          this.pending.delete(response.request_id);
          pending.resolve(response);
          return;
        }
      }
    } catch {
      // Diagnostic lines are not protocol responses.
    }
    this.diag(`bridge: ${trimmed}`);
  }

  async query(query, topK = 3, timeoutMs = 800, signal = null, timelineMode = false) {
    const text = String(query || "").trim();
    if (!text) return null;
    // Never await the uv installer from pre-retrieval: fail open immediately
    // while the environment gate is still checking/provisioning.
    if (!["ready", "installed"].includes(this.runtime.state.status)) return null;
    const child = await this.ensure();
    if (!child?.stdin || child.stdin.destroyed) return null;
    const requestId = randomUUID();
    return new Promise((resolve) => {
      let settled = false;
      let timer = null;
      const finish = (value) => {
        if (settled) return;
        settled = true;
        if (timer) clearTimeout(timer);
        this.pending.delete(requestId);
        signal?.removeEventListener?.("abort", onAbort);
        resolve(value);
      };
      const onAbort = () => finish(null);
      const effectiveTimeout = this.coldStart ? 15000 : timeoutMs;
      this.coldStart = false;
      timer = setTimeout(() => finish(null), effectiveTimeout);
      if (signal) {
        if (signal.aborted) {
          finish(null);
          return;
        }
        signal.addEventListener("abort", onAbort, { once: true });
      }
      this.pending.set(requestId, { resolve: finish });
      try {
        child.stdin.write(
          `${JSON.stringify({
            op: "query",
            request_id: requestId,
            query: text,
            top_k: topK,
            timeline_mode: timelineMode,
          })}\n`,
        );
      } catch (error) {
        this.diag(`bridge query write failed: ${error?.stack || error}`);
        finish(null);
      }
    });
  }

  async writeTurn(user, assistant, extractedFact = null) {
    if (!["ready", "installed"].includes(this.runtime.state.status)) {
      this._log(
        "error",
        `writeTurn blocked: runtime status=${this.runtime.state.status}`,
      );
      return false;
    }
    const child = await this.ensure();
    if (!child?.stdin || child.stdin.destroyed) {
      this._log("error", "writeTurn blocked: bridge stdin unavailable");
      return false;
    }
    try {
      child.stdin.write(
        `${JSON.stringify({
          user_msg: user,
          ai_msg: assistant,
          ...(extractedFact ? { extracted_fact: extractedFact } : {}),
        })}\n`,
      );
      return true;
    } catch (error) {
      this.diag(`bridge turn write failed: ${error?.stack || error}`);
      return false;
    }
  }

  close() {
    this.closed = true;
    if (this.child && !this.child.killed) {
      try {
        this.child.kill();
      } catch {
        // Disposal must not fail.
      }
    }
    this.child = null;
    for (const pending of this.pending.values()) pending.resolve(null);
    this.pending.clear();
  }
}
