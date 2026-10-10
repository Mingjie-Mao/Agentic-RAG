import { execFileSync } from "node:child_process";
import path from "node:path";

export default async function setup() {
  // Recover only proven private test uploads after a previous process was killed.
  const root = path.resolve("..");
  execFileSync(
    path.join(root, ".venv/bin/python"),
    ["-m", "scripts.cleanup_browser_uploads", "--apply"],
    { cwd: root, stdio: "inherit" },
  );
  const base = process.env.RAG_BASE_URL ?? "http://127.0.0.1:8000";
  for (let i = 0; i < 60; i++) {
    try {
      const response = await fetch(base + "/api/health");
      if (response.ok && (await fetch(base + "/")).ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Local app is not ready. Start make run first.");
}
