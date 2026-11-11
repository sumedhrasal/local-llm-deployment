/**
 * Stage 3c: an agent running on your own model.
 *
 * Pi is a minimal coding agent (tools: read, write, edit, bash). Here we embed it
 * with its SDK and point it at the local vLLM server. The model decides which tools
 * to call; Pi runs them. We give it only read and write, confined to ./workspace
 * (see sandbox.ts).
 *
 *   npm start                                       # vLLM on :8000
 *   LLM_BASE_URL=http://localhost:11434/v1 npm start   # or Ollama
 */
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
	createAgentSession,
	createReadToolDefinition,
	createWriteToolDefinition,
	ModelRuntime,
	SessionManager,
	SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { confinedReadOps, confinedWriteOps } from "./sandbox.ts";

const baseUrl = process.env.LLM_BASE_URL ?? "http://localhost:8000/v1";
const workspace = resolve("workspace");
const deploymentCard = resolve("../../deployment.json");

// 1. Ask the server which model it serves.
let modelId: string;
try {
	const res = await fetch(`${baseUrl}/models`);
	modelId = (await res.json()).data[0].id;
} catch {
	console.error(`Could not reach ${baseUrl}. Is the server running? Try: make deploy`);
	process.exit(1);
}

// 2. Tell Pi about it: a "provider" that speaks the OpenAI chat-completions API.
//    (The same JSON goes in ~/.pi/agent/models.json for the interactive `pi` CLI.)
const card = existsSync(deploymentCard) ? JSON.parse(readFileSync(deploymentCard, "utf8")) : {};
const isQwen3Thinking = card.reasoning_parser === "qwen3";
const providerConfig = {
	providers: {
		local: {
			baseUrl,
			api: "openai-completions",
			apiKey: "local", // vLLM doesn't check it, but Pi wants one
			models: [
				{
					id: modelId,
					contextWindow: card.max_model_len ?? 16384,
					maxTokens: 4096,
					reasoning: isQwen3Thinking,
					// Qwen3 turns thinking on/off through its chat template.
					...(isQwen3Thinking ? { compat: { thinkingFormat: "qwen-chat-template" } } : {}),
				},
			],
		},
	},
};
const configDir = mkdtempSync(join(tmpdir(), "pi-local-"));
writeFileSync(join(configDir, "models.json"), JSON.stringify(providerConfig, null, 2));

const modelRuntime = await ModelRuntime.create({
	modelsPath: join(configDir, "models.json"),
	authPath: join(configDir, "auth.json"),
});
const model = modelRuntime.getModel("local", modelId);
if (!model) throw new Error("model not registered");

// 3. Create an agent session: in memory, working inside ./workspace.
const { session } = await createAgentSession({
	cwd: workspace,
	agentDir: configDir, // don't read or write your personal ~/.pi settings
	model,
	modelRuntime,
	// Least privilege: this task needs to read and write files, nothing else.
	// Pi's defaults also include `bash` and `edit`; never hand an agent a shell it doesn't need.
	// The read/write tools we pass refuse any path outside ./workspace.
	noTools: "builtin",
	customTools: [
		createReadToolDefinition(workspace, { operations: confinedReadOps(workspace) }),
		createWriteToolDefinition(workspace, { operations: confinedWriteOps(workspace) }),
	],
	thinkingLevel: "off", // faster for a demo; try "medium"
	sessionManager: SessionManager.inMemory(workspace),
	settingsManager: SettingsManager.inMemory(),
});

// 4. Watch what the agent does.
session.subscribe((event) => {
	if (event.type === "message_update" && event.assistantMessageEvent.type === "text_delta") {
		process.stdout.write(event.assistantMessageEvent.delta);
	} else if (event.type === "tool_execution_start") {
		const args = JSON.stringify(event.args);
		console.log(`\n[tool] ${event.toolName} ${args.length > 100 ? `${args.slice(0, 100)}...` : args}`);
	} else if (event.type === "tool_execution_end" && event.isError) {
		console.log(`[tool] ${event.toolName} failed`);
	}
});

const task =
	"Read notes.md. Then write a summary of it to output/summary.md: exactly 5 bullet points, " +
	"one sentence each, aimed at a student new to LLM serving. Reply 'done' when the file is written.";

console.log(`Model: ${modelId} at ${baseUrl}`);
console.log(`Task:  ${task}\n`);
const started = Date.now();
try {
	await session.prompt(task);
} finally {
	session.dispose();
}
console.log(`\n\nFinished in ${((Date.now() - started) / 1000).toFixed(1)}s`);

const out = join(workspace, "output", "summary.md");
if (existsSync(out)) {
	console.log(`\n== ${out.replace(`${resolve("../..")}/`, "")} ==\n${readFileSync(out, "utf8")}`);
} else {
	console.log("\nThe agent did not write output/summary.md. Try running again, or a larger model.");
	process.exitCode = 1;
}
