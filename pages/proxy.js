// Forwards one request to the locally running system behind the Cloudflare tunnel.
// The app keeps every security rule (trial accounts, read-only, quotas); this only relays.
import ORIGIN from "./origin.js";

export function proxy({ request }, path) {
  const url = new URL(request.url);
  const target = new URL((path ?? url.pathname) + url.search, ORIGIN);
  return fetch(new Request(target, request), { redirect: "manual" }).catch(
    () => new Response("在线系统暂时不可用：本机服务或隧道未运行。", {
      status: 502,
      headers: { "content-type": "text/plain; charset=utf-8" },
    }),
  );
}
