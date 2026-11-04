/**
 * Model-facing memory_search tool registration.
 * The Python transport stays in bridge_process.js; this module only maps the
 * tool contract to that transport.
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

export function registerMemoryTools(ctx, { queryMemory, diag = () => {} }) {
  ctx.inject(["tools"], (child) => {
    child.effect(
      () =>
        child.tools.register(
          defineTool({
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
                required: false,
                description: "Maximum results, 1-20.",
              },
              timeline_mode: {
                type: "boolean",
                required: false,
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
              if (result?.ok === false) {
                diag(`memory_search dependency error: ${result.error || "unknown"}`);
                return { text: dependencyError(result) };
              }
              return {
                text: result?.ok
                  ? formatMemoryResults(result.result)
                  : "Memory search timed out or failed; continue without memory.",
              };
            },
          }),
        ),
      "DSH-DREAM.memory-search-tool",
    );
  });
}
