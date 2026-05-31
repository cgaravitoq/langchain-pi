// Node sidecar for langchain-pi. Wraps pi-ai streamSimple + pi-coding-agent
// registry/auth and streams normalized pi events as NDJSON, tagged with the
// request id. Node owns model resolution, OAuth refresh and the provider stealth
// headers — Python never touches a token. buildContext is ported verbatim from
// langchain-pi-ts/src/pi-conversions.ts so multi-turn history is byte-faithful.
import { readFileSync } from "node:fs";
import { resolve as resolvePath } from "node:path";
import { pathToFileURL } from "node:url";

const out = (o) => process.stdout.write(`${JSON.stringify(o)}\n`);

// LANGCHAIN_PI_NODE_MODULES resolves the pi packages by absolute URL (NODE_PATH
// is ignored for ESM); otherwise a bare import resolves them from an ancestor.
function entryUrl(nodeModules, name) {
  const dir = resolvePath(nodeModules, name);
  const pkg = JSON.parse(readFileSync(resolvePath(dir, "package.json"), "utf8"));
  const dot = pkg.exports?.["."];
  let entry = pkg.module ?? pkg.main ?? "index.js";
  if (typeof dot === "string") {
    entry = dot;
  } else if (dot) {
    const cond = dot.import ?? dot.node ?? dot.default;
    entry = typeof cond === "string" ? cond : (cond?.default ?? cond?.node ?? entry);
  }
  return pathToFileURL(resolvePath(dir, entry)).href;
}

let streamSimple;
let registry;
let initError;

const ready = (async () => {
  try {
    const nm = process.env.LANGCHAIN_PI_NODE_MODULES;
    const piAi = nm
      ? entryUrl(nm, "@earendil-works/pi-ai")
      : "@earendil-works/pi-ai";
    const piAgent = nm
      ? entryUrl(nm, "@earendil-works/pi-coding-agent")
      : "@earendil-works/pi-coding-agent";
    ({ streamSimple } = await import(piAi));
    const { AuthStorage, ModelRegistry } = await import(piAgent);
    registry = ModelRegistry.create(AuthStorage.create());
  } catch (e) {
    initError = String(e?.stack ?? e?.message ?? e);
  }
})();

const ZERO_USAGE = {
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

function buildContext(req, model) {
  const timestamp = Date.now();
  const messages = [];
  for (const m of req.messages ?? []) {
    if (m.role === "assistant") {
      const calls = (m.toolCalls ?? []).map((c) => ({
        type: "toolCall",
        id: c.id ?? "",
        name: c.name,
        arguments: c.arguments ?? {},
      }));
      messages.push({
        role: "assistant",
        content: [...(m.text ? [{ type: "text", text: m.text }] : []), ...calls],
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: ZERO_USAGE,
        stopReason: calls.length ? "toolUse" : "stop",
        timestamp,
      });
    } else if (m.role === "toolResult") {
      messages.push({
        role: "toolResult",
        toolCallId: m.toolCallId,
        toolName: m.toolName ?? "",
        content: [{ type: "text", text: m.text ?? "" }],
        isError: !!m.isError,
        timestamp,
      });
    } else {
      messages.push({ role: "user", content: m.text ?? "", timestamp });
    }
  }
  const ctx = { messages };
  if (req.systemPrompt) ctx.systemPrompt = req.systemPrompt;
  if (req.tools?.length) ctx.tools = req.tools;
  return ctx;
}

const controllers = new Map();

async function handle(req) {
  const id = req.id;
  await ready;
  if (initError) {
    out({ id, type: "error", error: { errorMessage: initError } });
    out({ id, type: "end" });
    return;
  }
  const controller = new AbortController();
  controllers.set(id, controller);
  try {
    const model = registry.find(req.provider, req.modelId);
    if (!model)
      return out({
        id,
        type: "error",
        error: { errorMessage: `Unknown model "${req.provider}/${req.modelId}"` },
      });

    const a = await registry.getApiKeyAndHeaders(model);
    if (!a.ok) return out({ id, type: "error", error: { errorMessage: a.error } });

    const stream = await streamSimple(model, buildContext(req, model), {
      reasoning: req.reasoning ?? "low",
      apiKey: a.apiKey,
      headers: a.headers,
      signal: controller.signal,
    });

    for await (const event of stream) {
      if (event.type === "text_delta") {
        out({ id, type: "text_delta", delta: event.delta });
      } else if (event.type === "toolcall_end") {
        const tc = event.toolCall;
        out({
          id,
          type: "toolcall_end",
          contentIndex: event.contentIndex,
          toolCall: { id: tc.id, name: tc.name, arguments: tc.arguments },
        });
      } else if (event.type === "done") {
        out({
          id,
          type: "done",
          stopReason: event.message.stopReason,
          usage: event.message.usage,
        });
      } else if (event.type === "error") {
        out({
          id,
          type: "error",
          error: {
            stopReason: event.error.stopReason,
            errorMessage: event.error.errorMessage,
          },
        });
      }
    }
  } catch (e) {
    out({ id, type: "error", error: { errorMessage: String(e?.message ?? e) } });
  } finally {
    controllers.delete(id);
    out({ id, type: "end" });
  }
}

// NDJSON: split on "\n" only (not U+2028/U+2029). Requests run sequentially.
let chain = Promise.resolve();
let buf = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  buf += chunk;
  let idx;
  while ((idx = buf.indexOf("\n")) >= 0) {
    const line = buf.slice(0, idx).replace(/\r$/, "");
    buf = buf.slice(idx + 1);
    if (!line.trim()) continue;
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      continue;
    }
    if (msg.type === "control" && msg.action === "abort") {
      controllers.get(msg.id)?.abort();
    } else {
      chain = chain.then(() => handle(msg));
    }
  }
});
process.stdin.on("end", () => {
  chain.then(() => process.stdout.write("", () => process.exit(0)));
});
