import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import { proxy } from "./proxy.js";
import { onRequest as liveShell } from "./functions/live/[[path]].js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

test("public HTTPS preserves separate cookies and adds Secure without changing expiry", async () => {
  const upstream = new Response("ok", { status: 200 });
  upstream.headers.append("Set-Cookie", "rag_session=fixture; Path=/; HttpOnly; SameSite=strict");
  upstream.headers.append("Set-Cookie", "extra=fixture; Expires=Wed, 21 Oct 2030 07:28:00 GMT; Secure");
  globalThis.fetch = async () => upstream;
  const response = await proxy({ request: new Request("https://example.test/api/auth/visitor") });
  const cookies = response.headers.getSetCookie();
  assert.equal(cookies.length, 2);
  assert.match(cookies[0], /; Secure$/);
  assert.match(cookies[0], /HttpOnly; SameSite=strict/);
  assert.equal(cookies[1], "extra=fixture; Expires=Wed, 21 Oct 2030 07:28:00 GMT; Secure");
  assert.equal(await response.text(), "ok");
});

test("HTTP preview keeps the upstream cookie policy", async () => {
  globalThis.fetch = async () => new Response("ok", { headers: { "Set-Cookie": "rag_session=fixture; HttpOnly" } });
  const response = await proxy({ request: new Request("http://localhost/api/auth/visitor") });
  assert.equal(response.headers.get("Set-Cookie"), "rag_session=fixture; HttpOnly");
});

test("API responses without cookies preserve status and body", async () => {
  globalThis.fetch = async () => new Response("denied", { status: 403 });
  const response = await proxy({ request: new Request("https://example.test/api/documents") });
  assert.equal(response.status, 403);
  assert.equal(response.headers.get("Set-Cookie"), null);
  assert.equal(await response.text(), "denied");
});

test("offline tunnel returns an explicit uncached error", async () => {
  globalThis.fetch = async () => { throw new Error("offline"); };
  const response = await proxy({ request: new Request("https://example.test/api/demo/status") });
  assert.equal(response.status, 502);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.match((await response.json()).detail, /真实执行回放/);
});

test("live shell uses static Pages assets and cannot cache an old login page", async () => {
  const response = await liveShell({
    request: new Request("https://example.test/live"),
    env: { ASSETS: { fetch: async (request) => {
      assert.equal(new URL(request.url).pathname, "/live/");
      return new Response("static demo", { headers: { "Content-Type": "text/html" } });
    } } },
  });
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(await response.text(), "static demo");
});
