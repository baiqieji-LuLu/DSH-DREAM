/**
 * Environment discovery and first-run bootstrapping for DSH-DREAM.
 *
 * This module owns every side effect that answers "which Python and which
 * Hugging Face cache should the bridge use?". index.js only assembles it.
 */
import { spawn } from "node:child_process";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { get as httpGet } from "node:http";
import { get as httpsGet } from "node:https";
import { dirname, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

export const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url));
export const PROJECT_ROOT = resolve(
  process.env.DSH_DREAM_ROOT || resolve(PLUGIN_DIR, ".."),
);
export const DSH_HOME = resolve(
  process.env.DSH_HOME ||
    resolve(process.env.USERPROFILE || process.env.HOME || ".", ".dsh"),
);
export const PROFILE_DIR = resolve(
  process.env.DSH_PROFILE_DIR || resolve(DSH_HOME, "profiles", "desktop"),
);
export const DREAM_VENV = resolve(
  process.env.DSH_DREAM_VENV || resolve(PROFILE_DIR, ".dream-venv"),
);
export const PIP_MIRROR =
  process.env.DSH_DREAM_PIP_MIRROR ||
  "https://pypi.tuna.tsinghua.edu.cn/simple";

export const ONNX_MODEL_DIR = resolve(
  PROFILE_DIR,
  ".dream-models",
  "bge-small-zh-v1.5",
);
export const ONNX_MODEL_FILE = resolve(ONNX_MODEL_DIR, "model.onnx");
export const ONNX_TOKENIZER_FILE = resolve(ONNX_MODEL_DIR, "tokenizer.json");
const ONNX_MODEL_URL =
  process.env.DSH_DREAM_ONNX_MODEL_URL ||
  "https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/onnx/model.onnx";
const ONNX_TOKENIZER_URL =
  process.env.DSH_DREAM_ONNX_TOKENIZER_URL ||
  "https://hf-mirror.com/Xenova/bge-small-zh-v1.5/resolve/main/tokenizer.json";
const ONNX_FALLBACK_BASE = "https://huggingface.co/Xenova/bge-small-zh-v1.5/resolve/main";

const RUNTIME_PROBE = [
  "import importlib.util,json,sys",
  "mods=('agent_memory','onnxruntime','tokenizers','sqlite_vec')",
  "missing=[m for m in mods if importlib.util.find_spec(m) is None]",
  "print(json.dumps({'ok':not missing,'missing':missing,'python':sys.executable}))",
].join("; ");

/** Verify a cache directory is actually writable before using it. */
function writableCacheDir(directory) {
  try {
    mkdirSync(directory, { recursive: true });
    const probe = resolve(
      directory,
      `.dsh-dream-write-probe-${process.pid}-${Date.now()}`,
    );
    writeFileSync(probe, "ok", "utf8");
    unlinkSync(probe);
    return directory;
  } catch {
    return null;
  }
}

function downloadFile(url, destination, redirects = 5) {
  return new Promise((resolvePromise, reject) => {
    if (existsSync(destination) && statSync(destination).size > 0) {
      resolvePromise(destination);
      return;
    }
    const temp = `${destination}.part`;
    const request = (url.startsWith("https:") ? httpsGet : httpGet)(
      url,
      { headers: { "User-Agent": "DSH-DREAM/0.5" } },
      (response) => {
        const status = response.statusCode || 0;
        if (status >= 300 && status < 400 && response.headers.location) {
          response.resume();
          if (redirects <= 0) {
            reject(new Error(`too many redirects: ${url}`));
            return;
          }
          const next = new URL(response.headers.location, url).toString();
          downloadFile(next, destination, redirects - 1).then(
            resolvePromise,
            reject,
          );
          return;
        }
        if (status !== 200) {
          response.resume();
          reject(new Error(`HTTP ${status} for ${url}`));
          return;
        }
        const stream = createWriteStream(temp);
        pipeline(response, stream)
          .then(() => {
            renameSync(temp, destination);
            resolvePromise(destination);
          })
          .catch((error) => {
            try {
              unlinkSync(temp);
            } catch {
              // Best-effort cleanup.
            }
            reject(error);
          });
      },
    );
    request.setTimeout(5 * 60 * 1000, () => {
      request.destroy(new Error(`download timed out: ${url}`));
    });
    request.on("error", (error) => {
      try {
        unlinkSync(temp);
      } catch {
        // Best-effort cleanup.
      }
      reject(error);
    });
  });
}

/** Download BGE ONNX assets into the profile, with a mirror and HF fallback. */
export async function ensureOnnxAssets(diag = () => {}) {
  if (
    existsSync(ONNX_MODEL_FILE) &&
    statSync(ONNX_MODEL_FILE).size > 0 &&
    existsSync(ONNX_TOKENIZER_FILE) &&
    statSync(ONNX_TOKENIZER_FILE).size > 0
  ) {
    return { ok: true, downloaded: false, dir: ONNX_MODEL_DIR };
  }
  try {
    mkdirSync(ONNX_MODEL_DIR, { recursive: true });
    diag(`[DSH-DREAM] downloading ONNX assets to ${ONNX_MODEL_DIR}`);
    await downloadFile(ONNX_MODEL_URL, ONNX_MODEL_FILE);
    await downloadFile(ONNX_TOKENIZER_URL, ONNX_TOKENIZER_FILE);
    return { ok: true, downloaded: true, dir: ONNX_MODEL_DIR };
  } catch (mirrorError) {
    try {
      diag(`[DSH-DREAM] mirror failed, trying huggingface.co: ${mirrorError}`);
      await downloadFile(
        `${ONNX_FALLBACK_BASE}/onnx/model.onnx`,
        ONNX_MODEL_FILE,
      );
      await downloadFile(
        `${ONNX_FALLBACK_BASE}/tokenizer.json`,
        ONNX_TOKENIZER_FILE,
      );
      return { ok: true, downloaded: true, dir: ONNX_MODEL_DIR };
    } catch (hfError) {
      return {
        ok: false,
        downloaded: false,
        dir: ONNX_MODEL_DIR,
        error: String(hfError),
      };
    }
  }
}

/** Resolve HF_HOME in priority order, failing over when a path is read-only. */
export function resolveCache(configured = "") {
  const candidates = [
    process.env.DSH_DREAM_HF_CACHE,
    configured,
    resolve(PROFILE_DIR, ".hf-cache"),
    resolve(PROJECT_ROOT, ".hf-cache"),
    resolve(process.cwd(), ".hf-cache"),
  ];
  for (const candidate of candidates) {
    const text = String(candidate || "").trim();
    if (!text) continue;
    const writable = writableCacheDir(resolve(text));
    if (writable) return { path: writable, error: "" };
  }
  return {
    path: null,
    error: "模型缓存目录不可写，请手动在设置中指定目录",
  };
}

/**
 * Resolve the Python interpreter:
 * env override -> UI override -> .dream-venv -> .venv -> PATH.
 */
export function resolvePython(configured = "") {
  const explicit = String(
    process.env.DSH_DREAM_PYTHON || configured || "",
  ).trim();
  if (explicit) return explicit;

  const roots = [
    process.env.DSH_DREAM_ROOT,
    PROJECT_ROOT,
    PROFILE_DIR,
    process.cwd(),
  ].filter(Boolean);
  const venvPaths =
    process.platform === "win32"
      ? [
          ".dream-venv\\Scripts\\python.exe",
          ".venv\\Scripts\\python.exe",
        ]
      : [".dream-venv/bin/python", ".venv/bin/python"];

  for (const root of new Set(roots)) {
    for (const relative of venvPaths) {
      const candidate = resolve(root, relative);
      if (existsSync(candidate)) return candidate;
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

export function venvPythonPath() {
  return process.platform === "win32"
    ? resolve(DREAM_VENV, "Scripts", "python.exe")
    : resolve(DREAM_VENV, "bin", "python");
}

/** Check the selected interpreter and cache without opening Chroma. */
export function probeRuntime(pythonPath, configuredCache = "", timeoutMs = 5000) {
  return new Promise((resolve) => {
    const python = resolvePython(pythonPath);
    const cache = resolveCache(configuredCache);
    if (!cache.path) {
      resolve({ ok: false, error: cache.error, cache });
      return;
    }
    let child = null;
    let settled = false;
    let timer = null;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      if (child && !child.killed) {
        try {
          child.kill();
        } catch {
          // Probe cleanup must not fail the request.
        }
      }
      resolve(value);
    };
    try {
      child = spawn(python, ["-c", RUNTIME_PROBE], {
        cwd: PROJECT_ROOT,
        env: {
          ...process.env,
          DSH_DREAM_PYTHON: python,
          DSH_DREAM_HF_CACHE: cache.path,
          HF_HOME: cache.path,
          HF_HUB_CACHE: resolve(cache.path, "hub"),
        },
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      });
    } catch (error) {
      finish({ ok: false, error: String(error), cache });
      return;
    }
    let stdout = "";
    let stderr = "";
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => finish({ ok: false, error: String(error), cache }));
    child.on("close", (code) => {
      try {
        const lines = stdout.trim().split(/\r?\n/).filter(Boolean);
        const payload = JSON.parse(lines.at(-1) || "{}");
        finish({
          ok: payload.ok === true,
          missing: Array.isArray(payload.missing) ? payload.missing : [],
          python: payload.python || python,
          code,
          cache,
          error: payload.ok === true ? "" : stderr.trim(),
        });
      } catch (error) {
        finish({
          ok: false,
          error: `${stderr.trim() || String(error)}`,
          code,
          cache,
        });
      }
    });
    timer = setTimeout(
      () => finish({ ok: false, error: "runtime probe timed out", cache }),
      timeoutMs,
    );
  });
}

function runProcess(command, args, extraEnv = {}) {
  return new Promise((resolve, reject) => {
    let child = null;
    try {
      child = spawn(command, args, {
        cwd: PROJECT_ROOT,
        env: { ...process.env, ...extraEnv },
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (error) {
      reject(error);
      return;
    }
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {
        // Ignore cleanup failure.
      }
      reject(new Error(`provisioning timed out: ${command} ${args.join(" ")}`));
    }, 30 * 60 * 1000);
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else {
        reject(
          new Error(
            `${command} exited ${code}: ${(stderr || stdout).trim().slice(-800)}`,
          ),
        );
      }
    });
  });
}

/**
 * Global environment state machine. `ensureReady()` is the single gate used by
 * both warmup and pre-retrieval so two callers can never initialize ChromaDB
 * concurrently.
 */
export class RuntimeEnv {
  constructor({ target, readBridgeFile, diag = () => {} }) {
    this.target = target;
    this.readBridgeFile = readBridgeFile;
    this.diag = diag;
    this.ready = null;
    this.state = {
      status: "idle",
      python: "",
      venv: DREAM_VENV,
      error: "",
    };
  }

  _config() {
    try {
      return this.readBridgeFile(this.target) || {};
    } catch {
      return {};
    }
  }

  async _ensureOnnxAssets() {
    return ensureOnnxAssets((message) => {
      try {
        this.diag(message);
      } catch {
        // Diagnostics must never block provisioning.
      }
    });
  }

  async ensureReady() {
    if (this.ready) return this.ready;
    this.ready = this._ensureReady().catch((error) => {
      this.state = {
        ...this.state,
        status: "error",
        error: String(error),
      };
      return this.state;
    });
    return this.ready;
  }

  async _ensureReady() {
    const bridgeConfig = this._config();
    const cache = resolveCache(bridgeConfig.hf_home_path);
    if (!cache.path) {
      this.state = {
        status: "error",
        python: resolvePython(bridgeConfig.python_path),
        venv: DREAM_VENV,
        error: cache.error,
      };
      return this.state;
    }
    this.state = {
      status: "checking",
      python: resolvePython(bridgeConfig.python_path),
      venv: DREAM_VENV,
      error: "",
    };
    const probe = await probeRuntime(
      bridgeConfig.python_path,
      bridgeConfig.hf_home_path,
    );
    if (probe.ok) {
      const assets = await this._ensureOnnxAssets();
      this.state = {
        status: "ready",
        python: probe.python || this.state.python,
        venv: DREAM_VENV,
        error: "",
        warning: assets.ok
          ? ""
          : "ONNX 资产下载失败，正在下载备用引擎（sentence-transformers）",
      };
      return this.state;
    }
    if (process.env.DSH_DREAM_AUTO_PROVISION === "0") {
      this.state = {
        status: "error",
        python: probe.python || this.state.python,
        venv: DREAM_VENV,
        error: "auto-provisioning disabled by DSH_DREAM_AUTO_PROVISION=0",
      };
      return this.state;
    }

    this.state = {
      ...this.state,
      status: "provisioning",
      error: "",
    };
    const uv = String(process.env.DSH_DREAM_UV || "uv").trim();
    try {
      if (!existsSync(venvPythonPath())) {
        await runProcess(uv, ["venv", DREAM_VENV]);
      }
      await runProcess(
        uv,
        [
          "pip",
          "install",
          "--python",
          venvPythonPath(),
          "-e",
          PROJECT_ROOT,
          "onnxruntime",
          "tokenizers",
          "sqlite-vec",
          "rank-bm25",
          "jinja2",
          "numpy<2",
          "--quiet",
          "--index-url",
          PIP_MIRROR,
        ],
        {
          DSH_DREAM_HF_CACHE: cache.path,
          HF_HOME: cache.path,
          UV_HTTP_TIMEOUT: "120",
        },
      );
      const assets = await this._ensureOnnxAssets();
      this.state = {
        status: "installed",
        python: venvPythonPath(),
        venv: DREAM_VENV,
        error: "",
        warning: assets.ok
          ? ""
          : "ONNX 资产下载失败，正在下载备用引擎（sentence-transformers）",
      };
    } catch (error) {
      this.state = {
        status: "error",
        python: this.state.python,
        venv: DREAM_VENV,
        error: String(error),
      };
    }
    return this.state;
  }
}
