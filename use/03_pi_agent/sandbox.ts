/**
 * Keep an agent's file tools inside one directory.
 *
 * Pi's read/write tools accept pluggable file operations, and every operation
 * receives the absolute path the model asked for. We resolve that path ourselves
 * (following `..` and symlinks), refuse anything that lands outside the workspace,
 * and then read or write the *resolved* path, opened with O_NOFOLLOW.
 *
 * This confines file access only. For full isolation, run the agent in a
 * container or VM.
 */
import { constants } from "node:fs";
import { access, lstat, mkdir, open, realpath } from "node:fs/promises";
import { basename, dirname, resolve, sep } from "node:path";

const denied = (path: string, why = "is outside the workspace") => new Error(`Access denied: ${path} ${why}`);

/** Resolve symlinks for the part of `path` that exists; the rest must be genuinely missing. */
async function resolvePath(path: string): Promise<string> {
	let current = resolve(path);
	const missing: string[] = [];
	for (;;) {
		try {
			return resolve(await realpath(current), ...missing.reverse());
		} catch (e) {
			if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e; // fail closed (ELOOP, EACCES, ...)
			// realpath says "not found", but a dangling symlink is "found" by lstat:
			// writing to it would create its (possibly outside) target.
			const exists = await lstat(current).then(() => true, () => false);
			if (exists) throw denied(path, "is a symlink to a missing target");
			const parent = dirname(current);
			if (parent === current) throw denied(path);
			missing.push(basename(current));
			current = parent;
		}
	}
}

/** The resolved path, if it is inside `root`; otherwise throws. */
export async function assertInside(root: string, path: string): Promise<string> {
	const realRoot = await realpath(root);
	const real = await resolvePath(path);
	if (real !== realRoot && !real.startsWith(realRoot + sep)) throw denied(path);
	return real;
}

export function confinedReadOps(root: string) {
	return {
		readFile: async (p: string) => {
			const f = await open(await assertInside(root, p), constants.O_RDONLY | constants.O_NOFOLLOW);
			try {
				return await f.readFile();
			} finally {
				await f.close();
			}
		},
		access: async (p: string) => access(await assertInside(root, p)),
	};
}

export function confinedWriteOps(root: string) {
	return {
		writeFile: async (p: string, content: string) => {
			const flags = constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC | constants.O_NOFOLLOW;
			const f = await open(await assertInside(root, p), flags, 0o644);
			try {
				await f.writeFile(content);
			} finally {
				await f.close();
			}
		},
		mkdir: async (dir: string) => {
			await mkdir(await assertInside(root, dir), { recursive: true });
		},
	};
}
