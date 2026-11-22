/**
 * Recall telemetry tests (no DSH runtime).
 *
 * Run with:  node tests/recall_telemetry.test.mjs
 */

import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join, resolve } from "node:path";
import {
  appendRecallTelemetry,
  resolveRecallPath,
} from "../dsh_plugin/recall_telemetry.js";

const testRoot = join(fileURLToPath(new URL("..", import.meta.url)), ".test-tmp");
mkdirSync(testRoot, { recursive: true });
const temp = mkdtempSync(join(testRoot, "recall-telemetry-"));
try {
  assert.equal(
    resolveRecallPath({
      env: { DSH_DREAM_RECALL_LOG: join(temp, "custom.jsonl") },
      home: temp,
    }),
    resolve(temp, "custom.jsonl"),
  );
  assert.equal(
    resolveRecallPath({
      env: { DSH_HOME: join(temp, "dsh-home") },
      home: temp,
    }),
    resolve(join(temp, "dsh-home"), "dream", "recall.jsonl"),
  );

  const logPath = join(temp, "recall.jsonl");
  const records = [
    { ts: "2026-11-22T00:00:00.000Z", turn: 1, gate: { outcome: "inject" } },
    { ts: "2026-11-22T00:00:01.000Z", turn: 2, gate: { outcome: "skip" } },
    { ts: "2026-11-22T00:00:02.000Z", turn: 3, gate: { outcome: "inject" } },
  ];
  for (const record of records) {
    assert.equal(appendRecallTelemetry(record, { path: logPath }), true);
  }
  const lines = readFileSync(logPath, "utf8").trim().split("\n");
  assert.equal(lines.length, 3);
  assert.deepEqual(JSON.parse(lines[2]), records[2]);

  // Rotation: a tiny limit moves the previous file to `.1`.
  const rotatePath = join(temp, "rotate.jsonl");
  assert.equal(
    appendRecallTelemetry({ n: 1 }, { path: rotatePath, maxBytes: 1 }),
    true,
  );
  assert.equal(
    appendRecallTelemetry({ n: 2 }, { path: rotatePath, maxBytes: 1 }),
    true,
  );
  assert.equal(existsSync(`${rotatePath}.1`), true);
  assert.equal(JSON.parse(readFileSync(rotatePath, "utf8").trim()).n, 2);

  // Write failure must be swallowed and reported to onError.
  let reported = null;
  const ok = appendRecallTelemetry(
    { n: 3 },
    { path: temp, onError: (error) => (reported = error) },
  );
  assert.equal(ok, false);
  assert.ok(reported instanceof Error);

  console.log("recall_telemetry.test.mjs: 1 test passed");
} finally {
  rmSync(temp, { recursive: true, force: true });
}
