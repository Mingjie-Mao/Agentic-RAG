import { proxy } from "../../proxy.js";

// The app is a single page with no client-side routes: /live and anything under it
// serve its index.html, whose /assets and /api requests are relayed by sibling routes.
export const onRequest = (context) => proxy(context, "/");
