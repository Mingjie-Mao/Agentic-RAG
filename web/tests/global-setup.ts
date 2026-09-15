export default async function setup() {
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
