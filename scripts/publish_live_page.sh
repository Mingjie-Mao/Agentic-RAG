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
if [ "$(curl -s -m 15 -o /dev/null -w '%{http_code}' "$TUNNEL/api/demo/status")" != "200" ]; then
  echo "隧道没有响应，未发布：$TUNNEL" >&2; exit 1
fi
# The showcase is the working-tree site/; /live is added by the functions in pages/.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$ROOT/.runtime/pages-live"
rm -rf "$STAGE" && mkdir -p "$STAGE"
npm --prefix "$ROOT/web" run build
cp -R "$ROOT/site" "$STAGE/site"
mkdir -p "$STAGE/site/live"
cp "$ROOT/web/dist/index.html" "$STAGE/site/live/index.html"
for ASSET in assets cmaps standard_fonts; do
  cp -R "$ROOT/web/dist/$ASSET" "$STAGE/site/$ASSET"
done
# Render benchmark tables at publish time, so JavaScript/network failure cannot erase them.
"$ROOT/.venv/bin/python" "$ROOT/scripts/prerender_site.py" --site "$STAGE/site"
"$ROOT/.venv/bin/python" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); (p/"release.json").write_text(json.dumps({"revision":sys.argv[2],"built_at":sys.argv[3]}))' "$STAGE/site" "$(git -C "$ROOT" rev-parse --short HEAD)-working-tree" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'export default "%s";\n' "$TUNNEL" > "$ROOT/pages/origin.js"
cd "$ROOT/pages"
npx --prefix "$ROOT" wrangler pages deploy "$STAGE/site" --project-name agentic-rag --branch main --commit-dirty=true
