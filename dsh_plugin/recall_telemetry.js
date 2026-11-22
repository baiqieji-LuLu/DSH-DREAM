/**
 * Append-only recall telemetry.
 *
 * One JSON object per line is appended to `~/.dsh/dream/recall.jsonl` (or
 * `DSH_DREAM_RECALL_LOG`). The file is rotated to `.1` once it reaches the size
 * limit, so monitoring never has to read an unbounded file.
 *
 * Telemetry is best-effort: every filesystem failure is swallowed (reported to
 * `onError` when supplied) because it must never affect the agent pre-step.
 */
import {
  appendFileSync,
  existsSync,
  mkdirSync,
  renameSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";

export const DEFAULT_RECALL_LOG_MAX_BYTES = 5 * 1024 * 1024;

/**
 * Resolve the recall telemetry path.
 *
 * @param {{env?: Record<string, string|undefined>, home?: string}} [options]
 * @returns {string} absolute path
 */
export function resolveRecallPath({
  env = typeof process !== "undefined" ? process.env : {},
  home = homedir(),
} = {}) {
  const override = String(env.DSH_DREAM_RECALL_LOG || "").trim();
  if (override) return resolve(override);
  const dshHome = String(env.DSH_HOME || "").trim();
  const base = dshHome ? resolve(dshHome) : resolve(home, ".dsh");
  return resolve(base, "dream", "recall.jsonl");
}

/**
 * Append one recall record. Never throws.
 *
 * @param {object} record JSON-serializable recall record.
 * @param {{path?: string, maxBytes?: number, onError?: (error: Error) => void}} [options]
 * @returns {boolean} whether the record was written.
 */
export function appendRecallTelemetry(
  record,
  {
    path,
    maxBytes = DEFAULT_RECALL_LOG_MAX_BYTES,
    onError = () => {},
  } = {},
) {
  try {
    const target = path ? resolve(path) : resolveRecallPath();
    mkdirSync(dirname(target), { recursive: true });
    const limit = Number(maxBytes);
    if (
      Number.isFinite(limit) &&
      limit > 0 &&
      existsSync(target) &&
      statSync(target).size >= limit
    ) {
      const rotated = `${target}.1`;
      if (existsSync(rotated)) unlinkSync(rotated);
      renameSync(target, rotated);
    }
    appendFileSync(target, `${JSON.stringify(record)}\n`, "utf8");
    return true;
  } catch (error) {
    try {
      onError(error);
    } catch {
      // Diagnostics must never be the reason the gate fails.
    }
    return false;
  }
}
