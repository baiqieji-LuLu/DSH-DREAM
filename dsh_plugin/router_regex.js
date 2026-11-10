/**
 * Regex guard and last-resort arbiter for the v0.4.0 RouterEngine.
 *
 * Regex is fast and deterministic, but it is not a classifier: it only
 * recognises textual cues.  It has three jobs:
 *
 * 1. hard skip: empty input, greetings, slash commands, suppression phrases;
 * 2. strong explicit cues ("上次", "还记得", "previously", …): the only cues
 *    allowed to bypass the models when `regex.explicit_fast_path=true`;
 * 3. weak cues ("现在", "那个东西", "记得", "变化", …): **never** decide on
 *    their own.  They are handed to the local BGE head so semantics, not a
 *    keyword list, decides; regex only remains the final fallback.
 *
 * The engine only ever consumes `decision` from this module; it never asks the
 * regex layer to explain or to write prompt text.
 */

/** Rule-set version recorded in audit records; bump on behaviour changes. */
export const RULES_VERSION = 2;

export const HARD_SKIP = "hard_skip";
export const EXPLICIT = "explicit";
export const WEAK = "weak";
export const NEUTRAL = "neutral";

const GREETINGS = new Set([
  "你好",
  "您好",
  "早上好",
  "早安",
  "中午好",
  "下午好",
  "晚上好",
  "晚安",
  "在吗",
  "在不在",
  "嗨",
  "哈喽",
  "hi",
  "hello",
  "hey",
  "thanks",
  "thankyou",
  "谢谢",
]);

const COMMAND = /^\/(?:clear|new|reset|help|compact|status)\b/iu;

const SUPPRESSION_PATTERNS = [
  /别翻旧账/u,
  /不要翻旧账/u,
  /别(?:再)?提(?:以前|过去|从前|历史)/u,
  /不用(?:回忆|回想|检索|查记忆|提以前)/u,
  /(?:忽略|忘记|清空)(?:掉)?(?:记忆|长期记忆|历史)/u,
  /(?:don'?t|do not)\s+(?:recall|mention)\s+(?:the\s+)?(?:past|history)/iu,
];

/**
 * Strong explicit memory cues: the only patterns allowed to bypass the local /
 * cloud judge (and only when `explicit_fast_path=true`).
 */
const STRONG_PATTERNS = [
  /上次/u,
  /上回/u,
  /以前/u,
  /之前/u,
  /曾经/u,
  /当时/u,
  /那次/u,
  /那天/u,
  /后来/u,
  /还记得/u,
  /说过/u,
  /提过/u,
  /最早/u,
  /第一次/u,
  /最后一次/u,
  /\blast\s+time\b/iu,
  /\bpreviously\b/iu,
  /\bused\s+to\b/iu,
  /\bremember\b/iu,
  /\bback\s+then\b/iu,
];

/**
 * Weak cues: hints that the turn *may* need memory, but not proof.  They must
 * never short-circuit the local BGE head; a lexical match like "现在" or
 * "那个东西" is exactly what a semantic model is for.
 */
const WEAK_PATTERNS = [
  /记得/u,
  /历史/u,
  /时间线/u,
  /过程/u,
  /变化/u,
  /\bhistory\b/iu,
  /现在/u,
  /刚刚/u,
  /刚才/u,
  /最近/u,
  /那个/u,
  /这个/u,
  /今天/u,
  /昨天/u,
  /明天/u,
];

const PUNCTUATION_ONLY =
  /^[\s\p{P}\p{S}\u3000\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65]+$/u;

const STRIP_FOR_GREETING = /[，。！？、；：,.!?;:\s~～]/gu;

/**
 * Normalize a query the same way the cache key and audit hash do.
 *
 * @param {unknown} text
 * @returns {string}
 */
export function normalizeQuery(text) {
  return String(text ?? "")
    .normalize("NFKC")
    .replace(/\s+/gu, " ")
    .trim();
}

/**
 * @param {string} text
 * @returns {boolean}
 */
export function isGreeting(text) {
  const normalized = normalizeQuery(text).toLowerCase();
  if (!normalized) return false;
  const stripped = normalized.replace(STRIP_FOR_GREETING, "");
  if (!stripped) return false;
  return GREETINGS.has(stripped);
}

/**
 * Classify the guard stage.
 *
 * @param {string} query
 * @returns {{kind: string, reason?: string, decision?: "skip"|"retrieve"}}
 */
export function guard(query) {
  const text = normalizeQuery(query);
  if (!text) return { kind: HARD_SKIP, decision: "skip", reason: "empty" };
  if (PUNCTUATION_ONLY.test(text)) {
    return { kind: HARD_SKIP, decision: "skip", reason: "punctuation_only" };
  }
  if (COMMAND.test(text)) {
    return { kind: HARD_SKIP, decision: "skip", reason: "command" };
  }
  if (isGreeting(text)) {
    return { kind: HARD_SKIP, decision: "skip", reason: "greeting" };
  }
  for (const pattern of SUPPRESSION_PATTERNS) {
    if (pattern.test(text)) {
      return { kind: HARD_SKIP, decision: "skip", reason: "suppression" };
    }
  }
  for (const pattern of STRONG_PATTERNS) {
    if (pattern.test(text)) {
      return {
        kind: EXPLICIT,
        strength: "strong",
        decision: "retrieve",
        reason: "explicit_cue",
      };
    }
  }
  for (const pattern of WEAK_PATTERNS) {
    if (pattern.test(text)) {
      // Weak cues deliberately carry no decision: the engine keeps going and
      // lets the local BGE head judge semantics.
      return { kind: WEAK, strength: "weak", reason: "weak_cue" };
    }
  }
  return { kind: NEUTRAL };
}

/**
 * Binary regex router.  `route()` is the last-resort arbiter; it never returns
 * `uncertain` and always yields a rule conclusion.
 */
export class RegexRouter {
  /**
   * @param {object} [options]
   * @param {boolean} [options.enabled]
   * @param {"retrieve"|"skip"} [options.defaultDecision]
   * @param {boolean} [options.explicitFastPath]
   */
  constructor({
    enabled = true,
    defaultDecision = "retrieve",
    explicitFastPath = false,
  } = {}) {
    this.kind = "regex";
    this.id = "regex";
    this.enabled = enabled !== false;
    this.defaultDecision = defaultDecision === "skip" ? "skip" : "retrieve";
    this.explicitFastPath = explicitFastPath === true;
    this.rulesVersion = RULES_VERSION;
  }

  /** @returns {boolean} */
  isAvailable() {
    return true;
  }

  /**
   * Guard stage: hard skips, explicit cues, or neutral.
   *
   * @param {string} query
   * @returns {{kind: string, reason?: string, decision?: "skip"|"retrieve"}}
   */
  guard(query) {
    return guard(query);
  }

  /**
   * Final arbiter used after every model layer abstained or failed.
   *
   * @param {string} query
   * @param {{ defaultDecision?: "retrieve"|"skip" }} [options]
   * @returns {{decision: "retrieve"|"skip", probabilityRetrieve: number,
   *   confidence: number, uncertain: boolean, reason: string}}
   */
  route(query, options = {}) {
    if (!this.enabled) {
      const decision = this.defaultDecision;
      return {
        decision,
        probabilityRetrieve: decision === "retrieve" ? 1 : 0,
        confidence: 1,
        uncertain: false,
        reason: "regex_disabled",
      };
    }
    const result = guard(query);
    if (result.kind === HARD_SKIP) {
      return {
        decision: "skip",
        probabilityRetrieve: 0,
        confidence: 1,
        uncertain: false,
        reason: result.reason,
      };
    }
    if (result.kind === EXPLICIT) {
      return {
        decision: "retrieve",
        probabilityRetrieve: 1,
        confidence: 1,
        uncertain: false,
        reason: "explicit_cue",
      };
    }
    const fallback =
      options.defaultDecision === "skip" || options.defaultDecision === "retrieve"
        ? options.defaultDecision
        : this.defaultDecision;
    const weak = result.kind === WEAK;
    return {
      decision: fallback,
      probabilityRetrieve: fallback === "retrieve" ? 1 : 0,
      confidence: 1,
      uncertain: false,
      reason: weak
        ? fallback === "retrieve"
          ? "weak_cue_default_retrieve"
          : "weak_cue_default_skip"
        : fallback === "retrieve"
          ? "default_retrieve"
          : "default_skip",
    };
  }

  /**
   * @returns {{enabled: boolean, rulesVersion: number, defaultDecision: string,
   *   explicitFastPath: boolean}}
   */
  status() {
    return {
      enabled: this.enabled,
      rulesVersion: this.rulesVersion,
      defaultDecision: this.defaultDecision,
      explicitFastPath: this.explicitFastPath,
    };
  }
}

export default RegexRouter;
