// Research only: replay each assistant turn's recorded prefix through the pi build that the
// harness runs (.runtime/pi) and save the exact Chat Completions body pi would send for it.
// Usage: node capture_requests.mjs <trajectory.jsonl> <out.jsonl> <model id> <reasoning effort|off> <max_tokens>
import fs from "node:fs";
import http from "node:http";
import { stream } from "../../.runtime/pi/packages/ai/dist/api/openai-completions.js";

const [trajPath, outPath, modelId, effort, maxTokens] = process.argv.slice(2);
const messages = fs.readFileSync(trajPath, "utf8").trim().split("\n").map((line) => JSON.parse(line).message);

// Mirrors harness/pi.py model_config(); only baseUrl points at the local capture server.
const model = {
	id: modelId, name: modelId, api: "openai-completions", provider: "session",
	baseUrl: "http://127.0.0.1:18099/v1", input: ["text"], reasoning: effort !== "off",
	contextWindow: 262144, maxTokens: Number(maxTokens),
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	thinkingLevelMap: Object.fromEntries(["minimal", "low", "medium", "high", "xhigh", "max"].map((k) => [k, k])),
	compat: {
		supportsStore: false, supportsDeveloperRole: false,
		maxTokensField: modelId.toLowerCase().includes("gpt-") ? "max_completion_tokens" : "max_tokens",
		thinkingFormat: "openai", supportsReasoningEffort: true,
		requiresReasoningContentOnAssistantMessages: modelId.toLowerCase().includes("deepseek"),
	},
};

let resolveBody;
const server = http.createServer((req, res) => {
	let data = "";
	req.on("data", (chunk) => (data += chunk));
	req.on("end", () => {
		res.writeHead(200, { "Content-Type": "text/event-stream" });
		res.write(`data: ${JSON.stringify({ id: "x", choices: [{ index: 0, delta: { content: "ok" }, finish_reason: "stop" }] })}\n\n`);
		res.end("data: [DONE]\n\n");
		resolveBody(data);
	});
});
await new Promise((resolve) => server.listen(18099, "127.0.0.1", resolve));

const out = fs.createWriteStream(outPath);
for (let k = 0; k < messages.length; k++) {
	if (messages[k].role !== "assistant") continue;
	const body = new Promise((resolve) => (resolveBody = resolve));
	const events = stream(model, { messages: messages.slice(0, k) }, {
		apiKey: "sandbox", maxTokens: Number(maxTokens), reasoningEffort: effort === "off" ? undefined : effort,
	});
	for await (const _event of events) {
		// drain
	}
	out.write(`${JSON.stringify({ index: k, body: JSON.parse(await body) })}\n`);
}
out.end();
server.close();
