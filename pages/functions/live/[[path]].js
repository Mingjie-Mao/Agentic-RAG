// Serve the built shell from Pages, even when the local model is offline.
export const onRequest = async ({ request, env }) => {
  const url = new URL("/live/", request.url);
  const asset = await env.ASSETS.fetch(new Request(url, request));
  const response = new Response(asset.body, asset);
  // Always read the current shell; versioned JS/CSS retain their own caching policy.
  response.headers.set("Cache-Control", "no-store");
  return response;
};
