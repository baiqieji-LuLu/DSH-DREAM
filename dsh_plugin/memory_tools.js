/**
 * Model-facing memory tools.
 *
 * The Python transport stays in bridge_process.js; this module only maps the
 * tool contracts to that transport.  Keep registration logs explicit: a silent
 * failed mount is indistinguishable from "the model chose not to call it".
 */
import { defineTool } from "@deepseek-ai/dsh-tools";

export function formatMemoryResults(result) {
  const hits = Array.isArray(result?.results) ? result.results : [];
  const lines = hits
    .slice(0, 20)
    .map((hit, index) => {
      const content = String(hit?.content || hit?.text || "").trim();
      return content ? `${index + 1}. ${content}` : "";
    })
    .filter(Boolean);
  return lines.length
    ? lines.join("\n\n")
    : "No relevant long-term memories found.";
}

function dependencyError(result) {
  return [
    "Python environment error:",
    result.error || "unknown error",
    result.detail || "",
    `Python: ${result.python || "unknown"}`,
    `HF_HOME: ${result.hf_home || "unknown"}`,
    result.hint || "",
  ]
    .filter(Boolean)
    .join("\n");
}

function buildSearchTool({ queryMemory, diag }) {
  return defineTool({
    name: "memory_search",
    description:
      "Search trusted long-term memory from previous conversations. Use timeline_mode=true for history or process questions; otherwise leave it false.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "Natural-language memory query.",
      },
      top_k: {
        type: "integer",
        description: "Maximum results, 1-20.",
      },
      timeline_mode: {
        type: "boolean",
        description:
          "Return chronological raw memories instead of collapsing state changes.",
      },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          text: { type: "string", required: true },
        },
      },
      render(_args, value) {
        return [
          {
            type: "text",
            text: String(value?.text || "No memory results."),
          },
        ];
      },
    },
    async execute(args, exec) {
      const query = String(args?.query || "").trim();
      if (!query) throw new Error("memory_search query must not be empty");
      const topK = Math.min(20, Math.max(1, Number(args?.top_k) || 5));
      const result = await queryMemory(
        query,
        topK,
        5000,
        exec?.signal,
        args?.timeline_mode === true,
      );
      if (!result || result.ok === false) {
        diag(
          `memory_search dependency error: ${
            result?.error || "bridge unavailable"
          }`,
        );
        throw new Error("Memory engine not ready");
      }
      return {
        text: formatMemoryResults(result.result),
      };
    },
  });
}

function buildAddTurnTool({ writeTurn, diag }) {
  return defineTool({
    name: "memory_add_turn",
    description:
      "Write one complete turn (user message, assistant message, optional fact) to long-term memory. Only call when holographic_mode is OFF; when it is ON the Host writes automatically.",
    parameters: {
      user_msg: {
        type: "string",
        required: true,
        description: "The user's complete original message.",
      },
      ai_msg: {
        type: "string",
        required: true,
        description: "The assistant's complete original reply.",
      },
      extracted_fact: {
        type: "string",
        description: "Optional concise fact label for the same turn.",
      },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          text: { type: "string", required: true },
        },
      },
      render(_args, value) {
        return [{ type: "text", text: String(value?.text || "Turn stored.") }];
      },
    },
    async execute(args) {
      const user = String(args?.user_msg || "").trim();
      const assistant = String(args?.ai_msg || "").trim();
      if (!user || !assistant) {
        throw new Error("memory_add_turn requires user_msg and ai_msg");
      }
      const ok = await writeTurn(
        user,
        assistant,
        String(args?.extracted_fact || "").trim() || null,
      );
      if (!ok) {
        diag("memory_add_turn write failed");
        return {
          text: "Turn was not stored: Python memory bridge is unavailable.",
        };
      }
      return { text: "Turn stored in long-term memory." };
    },
  });
}

/**
 * Register both model-facing tools.
 *
 * DSH's current plugin contract exposes `ctx.tools` when the plugin declares
 * `inject = ["tools"]`.  The legacy `ctx.inject` path is retained as a fallback
 * for older hosts, but it is intentionally not the primary path: a missed
 * dynamic injection leaves the model with no tools and no visible error.
 */
export function registerMemoryTools(
  ctx,
  { queryMemory, writeTurn, diag = () => {} },
) {
  const tools = [
    buildSearchTool({ queryMemory, diag }),
    buildAddTurnTool({ writeTurn, diag }),
  ];

  const register = (host, label) => {
    for (const tool of tools) {
      try {
        host.effect(
          () => host.tools.register(tool),
          `DSH-DREAM.${tool.name}-tool`,
        );
        console.log(`[Tools] ${tool.name} registered successfully`);
        diag(`[Tools] ${tool.name} registered successfully`);
      } catch (error) {
        console.error(`[Tools] failed to register ${tool.name}:`, error);
        diag(`[Tools] failed to register ${tool.name}: ${error?.stack || error}`);
      }
    }
    if (!tools.length) console.warn(`[Tools] no tools registered for ${label}`);
  };

  if (ctx?.tools && typeof ctx.tools.register === "function") {
    if (typeof ctx.effect !== "function") {
      console.error("[Tools] ctx.effect missing; cannot mount tool lifetime");
      return;
    }
    register(ctx, "direct");
    return;
  }

  if (typeof ctx?.inject === "function") {
    console.log("[Tools] tools service unavailable; using ctx.inject fallback");
    ctx.inject(["tools"], (child) => {
      console.log("[Tools] ctx.inject([tools]) callback fired");
      register(child, "injected");
    });
    return;
  }

  console.error("[Tools] no tools service or ctx.inject available");
}
