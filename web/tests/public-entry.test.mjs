import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  configFile: false,
  server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  optimizeDeps: { noDiscovery: true, include: [] },
});
after(() => server.close());
const {
  connectPublicSession,
  isPublicDemoPath,
  PublicDemoFallback,
  publicSessionResponse,
} = await server.ssrLoadModule("/src/PublicDemo.tsx");
const visitor = { id: "visitor", trial: { enabled: true, read_only: true } };

test("valid read-only session is reused without creating a new session", async () => {
  const result = await connectPublicSession(
    async () => visitor,
    async () => assert.fail("must preserve the existing session"),
  );
  assert.equal(result, visitor);
});

test("missing or expired session enters the restricted visitor automatically", async () => {
  let entries = 0;
  const result = await connectPublicSession(
    async () => {
      throw Object.assign(new Error("expired"), { status: 401 });
    },
    async () => {
      entries++;
      return visitor;
    },
  );
  assert.equal(result, visitor);
  assert.equal(entries, 1);
});

test("public entry replaces a writable identity with a read-only visitor", async () => {
  const result = await connectPublicSession(
    async () => ({ id: "admin", trial: { enabled: false } }),
    async () => visitor,
  );
  assert.equal(result, visitor);
  await assert.rejects(
    connectPublicSession(
      async () => ({ trial: { enabled: false } }),
      async () => ({ trial: { enabled: true, read_only: false } }),
    ),
    /演示访客会话/,
  );
});

test("offline backend cannot be reported as a successful live session", async () => {
  await assert.rejects(
    connectPublicSession(
      async () => {
        throw new Error("offline");
      },
      async () => {
        throw new Error("offline");
      },
    ),
    /offline/,
  );
});

test("offline page contains an interactive historical replay without a login form", () => {
  const html = renderToStaticMarkup(
    React.createElement(PublicDemoFallback, {
      connecting: false,
      onRetry() {},
    }),
  );
  assert.match(html, /真实历史回放/);
  assert.match(html, /不调用模型、不消耗额度/);
  assert.match(html, /src="\/replay.html\?embedded=1"/);
  assert.doesNotMatch(html, /<form|type="password"|登录工作空间|进入只读演示/);
});

test("only the public live route enables automatic visitor entry", () => {
  for (const path of ["/live", "/live/", "/live/example"])
    assert.equal(isPublicDemoPath(path), true);
  for (const path of ["/", "/live-admin", "/login"])
    assert.equal(isPublicDemoPath(path), false);
});

test("expired public session restores once and retries the rejected request once", async () => {
  let sends = 0,
    renewals = 0;
  const response = await publicSessionResponse(
    "/chat",
    true,
    async () => new Response("", { status: ++sends === 1 ? 401 : 200 }),
    async () => {
      renewals++;
    },
  );
  assert.equal(response.status, 200);
  assert.equal(sends, 2);
  assert.equal(renewals, 1);
});

test("auth, local sessions, quotas and model failures never trigger automatic replay", async () => {
  for (const [path, publicDemo, status] of [
    ["/auth/visitor", true, 401],
    ["/documents", false, 401],
    ["/chat", true, 403],
    ["/chat", true, 429],
    ["/chat", true, 500],
    ["/chat", true, 502],
    ["/chat", true, 503],
    ["/chat", true, 504],
  ]) {
    let sends = 0;
    const response = await publicSessionResponse(
      path,
      publicDemo,
      async () => {
        sends++;
        return new Response("", { status });
      },
      async () =>
        assert.fail("must not repeat an execution or alter local login"),
    );
    assert.equal(response.status, status);
    assert.equal(sends, 1);
  }
});

test("renewal failure and repeated rejection cannot create a retry loop", async () => {
  let sends = 0;
  const send = async () => {
    sends++;
    return new Response("", { status: 401 });
  };
  const rejected = await publicSessionResponse(
    "/documents",
    true,
    send,
    async () => {},
  );
  assert.equal(rejected.status, 401);
  assert.equal(sends, 2);
  sends = 0;
  await assert.rejects(
    publicSessionResponse("/documents", true, send, async () => {
      throw new Error("unavailable");
    }),
    /unavailable/,
  );
  assert.equal(sends, 1);
});
