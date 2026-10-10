import React from "react";
import { BookOpen, LoaderCircle } from "lucide-react";

export function isPublicDemoPath(path: string) {
  return /^\/live(?:\/|$)/.test(path);
}

export async function publicSessionResponse(
  path: string,
  publicDemo: boolean,
  send: () => Promise<Response>,
  renew: () => Promise<unknown>,
) {
  const response = await send();
  // Authentication fails before a request executes. Retry only that rejection, once.
  if (publicDemo && response.status === 401 && !path.startsWith("/auth/")) {
    await renew();
    return send();
  }
  return response;
}

type Visitor = { trial: { enabled: boolean; read_only?: boolean } };

// Keep server authorization; recover a bounded visitor session without a login UI.
export async function connectPublicSession<T extends Visitor>(
  current: () => Promise<T>,
  enter: () => Promise<T>,
): Promise<T> {
  try {
    const user = await current();
    if (user.trial.enabled && user.trial.read_only) return user;
  } catch {
    // Missing, expired, or unavailable session: try the restricted visitor entry.
  }
  const user = await enter();
  if (!user.trial.enabled || !user.trial.read_only)
    throw new Error("演示访客会话暂时不可用");
  return user;
}

export function PublicDemoFallback({
  connecting,
  onRetry,
}: {
  connecting: boolean;
  onRetry: () => void;
}) {
  return (
    <main className="public-demo-shell">
      <header className="public-demo-header">
        <a className="brand" href="/">
          <BookOpen size={23} /> 星桥知识库
        </a>
        <nav aria-label="项目链接">
          <a href="/">项目介绍</a>
          <a href="https://github.com/Mingjie-Mao/Agentic-RAG">GitHub ↗</a>
        </nav>
      </header>
      <section className="public-demo-status" aria-live="polite">
        <div>
          <h1>企业知识 Agent</h1>
          <p>星桥软件模拟企业环境 · 全部资料为虚构 · 无需登录。</p>
          <p role="status">
            {connecting
              ? "正在连接实时工作空间；你可以先查看下方真实执行回放。"
              : "实时服务暂时不可用，当前展示真实历史回放。恢复后将自动进入工作空间。"}
          </p>
          <small>回放不调用模型、不消耗额度，也不代表本次实时执行。</small>
        </div>
        <button className="secondary" disabled={connecting} onClick={onRetry}>
          {connecting ? <LoaderCircle className="spin" size={16} /> : null}
          {connecting ? "连接中…" : "重新连接实时服务"}
        </button>
      </section>
      <iframe
        className="public-demo-replay"
        title="真实历史执行回放"
        src="/replay.html?embedded=1"
      />
      <p className="public-demo-replay-link">
        <a href="/replay.html">在完整页面查看回放与原文 ↗</a>
      </p>
    </main>
  );
}
