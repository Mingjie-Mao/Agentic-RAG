// Forwards one request to the locally running system behind the Cloudflare tunnel.
// The app keeps every security rule (trial accounts, read-only, quotas); this only relays.
import ORIGIN from "./origin.js";

export function proxy({ request }, path) {
  const url = new URL(request.url);
  const target = new URL((path ?? url.pathname) + url.search, ORIGIN);
  return fetch(new Request(target, request), { redirect: "manual" }).then((response) => {
    if (url.protocol !== "https:") return response;
    // The local HTTP server may disable Secure for development. Public HTTPS
    // sessions must still be transport-restricted. Keep each cookie separate.
    const cookies = response.headers.getAll
      ? response.headers.getAll("Set-Cookie")
      : response.headers.getSetCookie();
    if (!cookies.length) return response;
    const secured = new Response(response.body, response);
    secured.headers.delete("Set-Cookie");
    for (const cookie of cookies) {
      secured.headers.append("Set-Cookie", /(?:^|;)\s*secure\s*(?:;|$)/i.test(cookie)
        ? cookie : `${cookie}; Secure`);
    }
    return secured;
  }).catch(
    () => new Response(JSON.stringify({ detail: "在线系统暂时不可用，可返回项目介绍查看真实执行回放。" }), {
      status: 502,
      headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
    }),
  );
}
