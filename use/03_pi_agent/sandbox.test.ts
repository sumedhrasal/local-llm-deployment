import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { assertInside, confinedReadOps, confinedWriteOps } from "./sandbox.ts";

const root = mkdtempSync(join(tmpdir(), "ws-"));
const outside = mkdtempSync(join(tmpdir(), "outside-"));
writeFileSync(join(root, "notes.md"), "hi");
writeFileSync(join(outside, "secret.txt"), "secret");
mkdirSync(join(root, "sub"));
symlinkSync(outside, join(root, "escape"));
symlinkSync(join(outside, "created-by-link.txt"), join(root, "dangling")); // target doesn't exist yet
symlinkSync(join(root, "loop"), join(root, "loop")); // ELOOP

test("allows files inside the workspace, existing or new", async () => {
	await assertInside(root, join(root, "notes.md"));
	await assertInside(root, join(root, "output", "new", "summary.md"));
	await assertInside(root, join(root, "sub"));
});

test("rejects absolute paths and .. outside the workspace", async () => {
	await assert.rejects(assertInside(root, "/etc/passwd"), /outside the workspace/);
	await assert.rejects(assertInside(root, join(root, "..", "x.txt")), /outside the workspace/);
	await assert.rejects(assertInside(root, `${root}-sibling/x`), /outside the workspace/);
});

test("rejects symlinks that point outside", async () => {
	await assert.rejects(assertInside(root, join(root, "escape", "secret.txt")), /outside the workspace/);
	await assert.rejects(assertInside(root, join(root, "escape", "new.txt")), /outside the workspace/);
});

test("read and write operations enforce the check", async () => {
	const r = confinedReadOps(root);
	assert.equal((await r.readFile(join(root, "notes.md"))).toString(), "hi");
	await assert.rejects(r.readFile(join(outside, "secret.txt")), /outside the workspace/);
	const w = confinedWriteOps(root);
	await w.mkdir(join(root, "output"));
	await w.writeFile(join(root, "output", "a.md"), "ok");
	await assert.rejects(w.writeFile(join(outside, "pwned.md"), "x"), /outside the workspace/);
	await assert.rejects(w.mkdir(join(outside, "d")), /outside the workspace/);
});

test("rejects a dangling symlink that would create a file outside", async () => {
	await assert.rejects(assertInside(root, join(root, "dangling")), /outside the workspace|symlink/);
	const w = confinedWriteOps(root);
	await assert.rejects(w.writeFile(join(root, "dangling"), "x"), /outside the workspace|symlink/);
	assert.equal(existsSync(join(outside, "created-by-link.txt")), false);
});

test("fails closed on errors other than 'not found'", async () => {
	await assert.rejects(assertInside(root, join(root, "loop", "x.txt")));
});
