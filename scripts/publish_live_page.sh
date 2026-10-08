#!/bin/sh
# Publish the showcase with a /live page relaying to the running system.
# Usage: scripts/publish_live_page.sh https://<name>.trycloudflare.com
# A quick tunnel gets a new address on every restart; rerun this with the new one.
set -eu
TUNNEL="${1:?tunnel URL required}"
# Refuse placeholders and dead tunnels: publishing either would break /live for everyone.
case "$TUNNEL" in
  https://*.trycloudflare.com) ;;
  *) echo "不是有效的 trycloudflare 地址：$TUNNEL" >&2; exit 1 ;;
esac
if ! printf '%s' "$TUNNEL" | grep -Eq '^https://[a-z0-9-]+\.trycloudflare\.com$'; then
  echo "地址含非法字符，请填写 cloudflared 输出的真实地址：$TUNNEL" >&2; exit 1
fi
if [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "$TUNNEL/api/demo/accounts")" != "200" ]; then
  echo "隧道没有响应，未发布：$TUNNEL" >&2; exit 1
fi
# The showcase is the working-tree site/; /live is added by the functions in pages/.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/.runtime/pages-live"
rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -R "$ROOT/site" "$STAGE/site"
printf 'export default "%s";\n' "$TUNNEL" > "$ROOT/pages/origin.js"
cd "$ROOT/pages"
npx --prefix "$ROOT" wrangler pages deploy "$STAGE/site" --project-name agentic-rag --branch main --commit-dirty=true
