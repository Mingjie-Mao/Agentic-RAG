import React, { useEffect, useRef, useState } from "react";
import type { PdfLocator } from "./PdfEvidence";
const PdfEvidence = React.lazy(() => import("./PdfEvidence"));
import { createRoot } from "react-dom/client";
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  Clock3,
  FileText,
  FolderOpen,
  History,
  Info,
  LoaderCircle,
  LogOut,
  Search,
  ShieldCheck,
  Upload,
  Workflow,
  X,
  ExternalLink,
  PanelRightClose,
} from "lucide-react";
import "./style.css";
import {
  connectPublicSession,
  isPublicDemoPath,
  PublicDemoFallback,
  publicSessionResponse,
} from "./PublicDemo";
import {
  agentExamples,
  chatExamples,
  modeLabels,
  AgentTimeline,
  SourceExcerpt,
} from "./DemoUX";

type User = {
  id: string;
  name: string;
  username: string;
  tenant: string;
  tenant_id: string;
  role: string;
  groups: string[];
  trial: TrialStatus;
};
type TrialStatus = {
  enabled: boolean;
  read_only?: boolean;
  used?: number;
  limit?: number;
  remaining?: number;
};
type Doc = {
  id: string;
  title: string;
  filename: string;
  status: string;
  error: string | null;
  chunks: number;
  tenant_public: boolean;
  groups: string[];
  version_id: string;
  media_type: string;
  timings: Record<string, number>;
};
type Citation = {
  chunk_id: string;
  document_id: string;
  version_id: string;
  title: string;
  locator: PdfLocator;
  text: string;
  original_url: string;
  preview_url: string;
};
type Result = {
  id: string;
  question: string;
  status: string;
  claims: { text: string; evidence_ids: string[]; quotes: string[] }[];
  verdict?: {
    value: "yes" | "no" | "unclear";
    claim_index: number | null;
  } | null;
  citations: Citation[];
  message: string;
  trace?: {
    demo_cache?: { hit: boolean; source_created_at?: string };
    method: string;
    scope?: { readable_documents: number; searchable_documents: number };
    original_question: string;
    retrieval_query: string;
    query_rewritten: boolean;
    embedding_model: string;
    generation_model: string;
    top_k: number;
    min_similarity: number;
    context_token_budget: number;
    context_tokens_estimate: number;
    candidates: {
      chunk_id: string;
      rank: number;
      title: string;
      locator_label: string;
      score: number;
      bm25_rank: number | null;
      dense_rank: number | null;
      fusion_score: number | null;
      admitted: boolean;
      excluded_because: string | null;
      evidence_id?: string;
    }[];
    embed_ms: number;
    retrieval_ms: number;
    generation_ms: number;
    total_ms: number;
  };
  usage?: { prompt_tokens: number; completion_tokens: number };
};
type Evidence = Citation & {
  media_type: string;
  filename: string;
  content_hash: string;
};
export type AgentEvent = {
  sequence: number;
  event_type: string;
  tool_name: string | null;
  payload: Record<string, unknown>;
  evidence_refs: string[];
  created_at: string;
};
export type AgentTask = {
  id: string;
  goal: string;
  mode: "auto" | "workflow" | "dynamic" | "hybrid" | "planner" | "adaptive";
  created_at: string;
  updated_at: string;
  status: string;
  step_no: number;
  max_steps: number;
  result: Omit<Result, "id" | "question">;
  error: string | null;
  events: AgentEvent[];
};

const publicDemo = isPublicDemoPath(location.pathname);
let visitorSession: Promise<User> | null = null;
function renewVisitorSession(): Promise<User> {
  if (!visitorSession) {
    visitorSession = api<User>("/auth/visitor", {
      method: "POST",
      signal: AbortSignal.timeout(6000),
    }).finally(() => {
      visitorSession = null;
    });
  }
  return visitorSession;
}

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const send = () =>
    fetch("/api" + path, {
      ...options,
      credentials: "same-origin",
      headers: {
        "X-Requested-With": "AgenticRAG",
        ...(options.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
        ...options.headers,
      },
    });
  const response = await publicSessionResponse(
    path,
    publicDemo,
    send,
    renewVisitorSession,
  );
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: "服务暂时不可用" }));
    const detail =
      typeof body.detail === "string" ? body.detail : "请求内容不符合要求";
    const failure = new Error(detail) as Error & {
      status?: number;
      retryable?: boolean;
      stage?: string;
      modelReached?: boolean;
    };
    failure.status = response.status;
    failure.retryable = [502, 503, 504].includes(response.status);
    failure.stage = typeof body.stage === "string" ? body.stage : undefined;
    failure.modelReached = body.model_reached === true;
    throw failure;
  }
  return response.json();
}

const statusLabels: Record<string, string> = {
  queued: "等待处理",
  processing: "正在入库",
  ready: "可检索",
  failed: "处理失败",
};
const groupLabels: Record<string, string> = {
  engineering: "工程组",
  support: "支持组",
};
const answerStates: Record<string, string> = {
  answered: "附原文依据",
  conflict: "附原文依据",
  access_changed: "权限已变化",
  verification_failed: "引用未通过检查",
  no_readable_documents: "没有可访问的资料",
  documents_processing: "资料处理中",
  insufficient_evidence: "依据不足",
};
const emptyStates: Record<string, { title: string; tone: string }> = {
  no_readable_documents: { title: "还没有你能访问的资料", tone: "neutral" },
  documents_processing: { title: "资料仍在处理", tone: "pending" },
  verification_failed: { title: "答案未通过引用核对", tone: "warn" },
};
function outageNote(failure: { stage?: string; modelReached?: boolean }) {
  if (failure.modelReached)
    return "（生成已开始，本次结果未知；重试会重新生成一次）";
  if (failure.stage === "retrieval" || failure.stage === "embedding")
    return "（失败发生在检索阶段，问题未提交给模型）";
  return "（本次结果未知，可以重试）";
}
const methodLabels: Record<string, string> = {
  dense: "向量检索",
  bm25: "关键词检索",
  hybrid: "混合检索（RRF 融合）",
};
const dropReasons: Record<string, string> = {
  below_min_similarity: "否 · 低于相似度阈值",
  context_budget_exhausted: "否 · 超出上下文预算",
};
function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [visitor, setVisitor] = useState<{
    visitor_enabled: boolean;
    daily_limit: number;
    notice: string;
  } | null>(null);
  const [loadingDemo, setLoadingDemo] = useState(true);
  const [accounts, setAccounts] = useState<User[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    api<{ visitor_enabled: boolean; daily_limit: number; notice: string }>(
      "/demo/status",
      { signal: AbortSignal.timeout(8000) },
    )
      .then(setVisitor)
      .catch(() => setError("在线服务暂时不可用，可先查看演示回放。"))
      .finally(() => setLoadingDemo(false));
    if (location.pathname.startsWith("/live")) return;
    api<{ accounts: User[]; password: string }>("/demo/accounts")
      .then((data) => {
        setAccounts(data.accounts);
        setUsername(
          data.accounts.find((a) => a.id === "xq-support")?.username ?? "",
        );
        setPassword(data.password);
      })
      .catch(() => {});
  }, []);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      onLogin(
        await api<User>("/auth/login", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function enterVisitor() {
    setBusy(true);
    setError("");
    try {
      onLogin(
        await api<User>("/auth/visitor", {
          method: "POST",
          signal: AbortSignal.timeout(15000),
        }),
      );
    } catch (e) {
      setError(
        (e as Error).name === "TimeoutError"
          ? "进入演示暂时超时，请重试或查看真实执行回放。"
          : (e as Error).message,
      );
    } finally {
      setBusy(false);
    }
  }
  const publicEntry =
    location.pathname.startsWith("/live") || visitor?.visitor_enabled;
  return (
    <main className="login-shell">
      <section className="login-story">
        <div className="brand">
          <BookOpen size={23} />
          <span>Agentic-RAG · 星桥知识库</span>
        </div>
        <div>
          <span className="eyebrow light">ENTERPRISE KNOWLEDGE</span>
          <h1>
            每个答案，
            <br />
            都有出处。
          </h1>
          <p>
            把散落在手册、制度和工单里的信息，
            <br />
            变成可以查证的工作答案。
          </p>
        </div>
        <div className="login-values">
          <span>
            <ShieldCheck size={18} /> 按权限查阅
          </span>
          <span>
            <FileText size={18} /> 回到原文
          </span>
        </div>
      </section>
      <section className="login-form-wrap">
        <form onSubmit={submit} className="login-form">
          <span className="eyebrow">开始查阅</span>
          <h2>{publicEntry ? "在线演示" : "进入你的工作空间"}</h2>
          {publicEntry ? (
            <>
              <p>星桥软件模拟企业环境。全部资料为虚构，无需账号或密码。</p>
              <div className="demo-note">
                <ShieldCheck size={18} />
                <span>
                  仅能查阅已授权资料；所有访客共享每天{" "}
                  {visitor?.daily_limit ?? "—"} 次问答 / Agent 额度，UTC
                  零点重置。
                </span>
              </div>
              <button
                type="button"
                className="primary login-button"
                onClick={enterVisitor}
                disabled={busy || loadingDemo || !visitor?.visitor_enabled}
              >
                {busy ? "正在进入…" : "在线体验"}
                <ChevronRight size={18} />
              </button>
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
              <div className="demo-links">
                <a href="/">返回项目介绍</a>
                <a href="/replay.html">查看真实执行回放</a>
                <a href="https://github.com/Mingjie-Mao/Agentic-RAG">
                  GitHub ↗
                </a>
              </div>
            </>
          ) : (
            <>
              <p>使用所属组织的账号登录。</p>
              <label>
                账号
                {accounts.length ? (
                  <select
                    aria-label="账号"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  >
                    {accounts.map((a) => (
                      <option key={a.id} value={a.username}>
                        {a.name} · {a.tenant}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    aria-label="账号"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                  />
                )}
              </label>
              <label>
                密码
                <input
                  aria-label="密码"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </label>
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
              <button className="primary login-button" disabled={busy}>
                {busy ? <LoaderCircle className="spin" size={18} /> : null}
                登录工作空间 <ChevronRight size={18} />
              </button>
              {accounts.length > 0 && (
                <div className="demo-note">
                  <Info size={16} />
                  <span>
                    本地演示使用虚构资料与示例账号。不同账号能访问的文档不同。
                  </span>
                </div>
              )}
            </>
          )}
        </form>
      </section>
    </main>
  );
}

function EvidencePane({
  evidence,
  onClose,
  loading = false,
  quotes = [],
  error = "",
  onRetry,
}: {
  error?: string;
  onRetry?: () => void;
  loading?: boolean;
  quotes?: string[];
  evidence: Evidence | null;
  onClose: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  if (loading)
    return (
      <aside className="evidence-pane">
        <div className="evidence-heading">
          来源与证据
          <button
            className="icon-button"
            aria-label="关闭证据"
            onClick={onClose}
          >
            <PanelRightClose size={18} />
          </button>
        </div>
        <p className="evidence-loading" role="status">
          <LoaderCircle className="spin" size={18} />
          正在加载引用并检查当前权限…
        </p>
      </aside>
    );
  if (error)
    return (
      <aside className="evidence-pane">
        <div className="evidence-heading">
          来源与证据
          <button
            className="icon-button"
            aria-label="关闭证据"
            onClick={onClose}
          >
            <PanelRightClose size={18} />
          </button>
        </div>
        <div className="evidence-content">
          <p role="alert">{error}</p>
          <p className="muted">原文尚未加载。重试时会再次检查当前访问权限。</p>
          <button className="secondary" onClick={onRetry}>
            重新加载引用
          </button>
        </div>
      </aside>
    );
  if (!evidence)
    return (
      <aside className="evidence-pane empty-evidence">
        <div className="evidence-heading">
          <span>来源与证据</span>
          <FileText size={18} />
        </div>
        <div className="empty-evidence-body">
          <div className="evidence-illustration">
            <FileText size={32} />
            <span>
              <Check size={14} />
            </span>
          </div>
          <h3>打开引用，核对原文</h3>
          <p>
            答案中的每条引用都能查看
            <br />
            原文片段与具体位置。
          </p>
          <div className="small-rule" />
          <span>资料始终在你的访问权限范围内</span>
        </div>
      </aside>
    );
  return (
    <aside className={`evidence-pane ${expanded ? "expanded" : ""}`}>
      <div className="evidence-heading">
        <span>来源与证据</span>
        <button
          className="secondary"
          onClick={() => setExpanded(!expanded)}
          aria-pressed={expanded}
        >
          {expanded ? "收起阅读" : "放大阅读"}
        </button>
        <button className="icon-button" aria-label="关闭证据" onClick={onClose}>
          <PanelRightClose size={18} />
        </button>
      </div>
      <div className="evidence-content">
        <span className="file-tag">
          {evidence.filename.split(".").pop()?.toUpperCase() ?? "文档"}
        </span>
        <h3>{evidence.title}</h3>
        <p className="locator">{evidence.locator.label}</p>
        {evidence.media_type === "application/pdf" && (
          <React.Suspense fallback={<p role="status">正在加载 PDF 阅读器…</p>}>
            <PdfEvidence
              url={evidence.original_url}
              locator={evidence.locator}
            />
          </React.Suspense>
        )}
        <a
          className="source-link"
          href={
            evidence.original_url +
            (evidence.locator.page ? `#page=${evidence.locator.page}` : "")
          }
          target="_blank"
          rel="noreferrer"
        >
          打开完整原文 <ExternalLink size={14} />
        </a>
        <SourceExcerpt text={evidence.text} quotes={quotes} />
        <details className="version-info">
          <summary>版本与来源标识</summary>
          <p>版本 {evidence.version_id}</p>
          <p>SHA-256 {evidence.content_hash}</p>
        </details>
      </div>
    </aside>
  );
}

function AnswerCard({
  result,
  onEvidence,
  showQuestion = true,
}: {
  showQuestion?: boolean;
  result: Result;
  onEvidence: (c: Citation, quotes: string[]) => void;
}) {
  const [trace, setTrace] = useState(false);
  const quotesFor = (c: Citation) =>
    result.claims
      .filter((claim) => claim.evidence_ids.includes(c.chunk_id))
      .flatMap((claim) => claim.quotes ?? []);
  return (
    <article className="answer-card" data-testid="answer-card">
      {showQuestion && <div className="answer-question">{result.question}</div>}
      <div className="answer-label">
        <span className="answer-logo">
          <BookOpen size={17} />
        </span>
        知识助手
        <span className="answer-status" data-testid="answer-status">
          {answerStates[result.status] ?? "依据不足"}
        </span>
      </div>
      {result.trace?.demo_cache?.hit && (
        <p className="scope-note" data-testid="cache-note">
          已核验答案缓存 · 已重新检查当前权限与资料版本 · 本次未调用模型。
          {result.trace.demo_cache.source_created_at &&
            ` 原答案生成于 ${new Date(result.trace.demo_cache.source_created_at).toLocaleString("zh-CN")}。`}
          缓存命中仍计入每日提问额度。
        </p>
      )}
      {result.status === "conflict" && (
        <p className="conflict-notice">{result.message}</p>
      )}
      {emptyStates[result.status] && (
        <div
          className={`empty-answer ${emptyStates[result.status].tone}`}
          data-testid="empty-answer"
        >
          <strong>{emptyStates[result.status].title}</strong>
          <p>{result.message}</p>
        </div>
      )}
      {result.trace?.scope && (
        <p className="scope-note" data-testid="scope-note">
          {result.trace.demo_cache?.hit
            ? "本次重新检查了你有权查看的 "
            : "本次只检索了你有权查看的 "}
          {result.trace.scope.readable_documents} 份资料
          {result.trace.scope.searchable_documents <
            result.trace.scope.readable_documents &&
            `，其中 ${result.trace.scope.searchable_documents} 份已可检索`}
          。
        </p>
      )}
      {result.verdict && result.claims.length ? (
        // A yes/no question deserves a yes/no. It is read out of the claims below,
        // so the reader can check it against the same cited sentences.
        <p
          className={`answer-verdict verdict-${result.verdict.value}`}
          data-testid="answer-verdict"
        >
          <strong>
            {
              {
                yes: "结论：是",
                no: "结论：否",
                unclear: "结论：依据不足以判断",
              }[result.verdict.value]
            }
          </strong>
          <span>
            根据下面第 {result.verdict.claim_index ?? "—"}{" "}
            条结论，原文引用见其后
          </span>
        </p>
      ) : null}
      {result.claims.length ? (
        <div className="answer-prose">
          {result.claims.map((claim, index) => (
            <p key={index}>
              {claim.text}{" "}
              <span className="inline-citations">
                {claim.evidence_ids.map((id) => {
                  const i = result.citations.findIndex(
                    (c) => c.chunk_id === id,
                  );
                  return i >= 0 ? (
                    <button
                      key={id}
                      aria-label={`引用 ${i + 1}`}
                      onClick={() =>
                        onEvidence(
                          result.citations[i],
                          quotesFor(result.citations[i]),
                        )
                      }
                    >
                      {i + 1}
                    </button>
                  ) : null;
                })}
              </span>
            </p>
          ))}
        </div>
      ) : !emptyStates[result.status] ? (
        <p className="no-answer">{result.message}</p>
      ) : null}
      {result.citations.length > 0 && (
        <div className="citation-grid">
          {result.citations.map((c, index) => (
            <button
              className="citation-card"
              key={c.chunk_id}
              onClick={() => onEvidence(c, quotesFor(c))}
            >
              <span className="citation-index">{index + 1}</span>
              <span>
                <strong>{c.title}</strong>
                <small>{c.locator.label}</small>
              </span>
              <ChevronRight size={15} />
            </button>
          ))}
        </div>
      )}
      {result.trace && (
        <>
          <button className="trace-toggle" onClick={() => setTrace(!trace)}>
            <Search size={13} />
            {trace ? "收起" : "查看"}检索过程
            <ChevronDown size={13} />
            <span>{(result.trace.total_ms / 1000).toFixed(1)} 秒</span>
          </button>
          {trace && (
            <div className="trace-panel" data-testid="trace-panel">
              <dl className="trace-query">
                <div>
                  <dt>原始问题</dt>
                  <dd>{result.trace.original_question}</dd>
                </div>
                <div>
                  <dt>实际检索问题</dt>
                  <dd>
                    {result.trace.retrieval_query}
                    {!result.trace.query_rewritten && <em>（未改写）</em>}
                  </dd>
                </div>
              </dl>
              <div className="trace-stats">
                <span>
                  检索方式{" "}
                  <b data-testid="trace-method">
                    {methodLabels[result.trace.method] ?? result.trace.method}
                  </b>
                </span>
                <span>
                  向量化 <b>{result.trace.embed_ms.toFixed(0)} ms</b>
                </span>
                <span>
                  检索 <b>{result.trace.retrieval_ms.toFixed(0)} ms</b>
                </span>
                <span>
                  生成 <b>{(result.trace.generation_ms / 1000).toFixed(1)} s</b>
                </span>
                <span>
                  上下文{" "}
                  <b>
                    {result.trace.context_tokens_estimate} /{" "}
                    {result.trace.context_token_budget} token
                  </b>
                </span>
                <span>
                  输入 / 输出{" "}
                  <b>
                    {result.usage?.prompt_tokens ?? "—"} /{" "}
                    {result.usage?.completion_tokens ?? "—"} token
                  </b>
                </span>
              </div>
              <p>
                {result.trace.embedding_model} → {result.trace.generation_model}{" "}
                · Top-K {result.trace.top_k}
                {result.trace.method === "dense" &&
                  ` · 相似度阈值 ${result.trace.min_similarity}`}
              </p>
              <table className="trace-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>资料</th>
                    <th>位置</th>
                    {result.trace.method === "hybrid" && (
                      <>
                        <th>关键词名次</th>
                        <th>向量名次</th>
                      </>
                    )}
                    <th>分数</th>
                    <th>是否作为证据</th>
                  </tr>
                </thead>
                <tbody>
                  {result.trace.candidates.map((c) => (
                    <tr
                      key={c.chunk_id}
                      className={c.admitted ? "admitted" : "dropped"}
                    >
                      <td>{c.rank}</td>
                      <td>{c.title}</td>
                      <td>{c.locator_label}</td>
                      {result.trace!.method === "hybrid" && (
                        <>
                          <td>{c.bm25_rank ?? "—"}</td>
                          <td>{c.dense_rank ?? "—"}</td>
                        </>
                      )}
                      <td>
                        <code>
                          {(c.fusion_score ?? c.score).toFixed(
                            c.fusion_score ? 5 : 3,
                          )}
                        </code>
                      </td>
                      <td>
                        {c.admitted ? (
                          <span className="tag-yes">是 · {c.evidence_id}</span>
                        ) : (
                          <span className="tag-no">
                            {dropReasons[c.excluded_because ?? ""] ?? "否"}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <small>
                分数只反映检索排序，不代表答案正确概率。混合检索按名次融合，不比较两路的原始分数。
              </small>
            </div>
          )}
        </>
      )}
    </article>
  );
}

function UploadDialog({
  user,
  onClose,
  onDone,
}: {
  user: User;
  onClose: () => void;
  onDone: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [scope, setScope] = useState("private");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append(
      "groups",
      JSON.stringify(scope === "private" || scope === "public" ? [] : [scope]),
    );
    form.append("tenant_public", String(scope === "public"));
    try {
      await api("/documents", { method: "POST", body: form });
      onDone();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="modal-backdrop">
      <form className="upload-modal" onSubmit={submit}>
        <div className="modal-heading">
          <h2>添加资料</h2>
          <button
            type="button"
            className="icon-button"
            aria-label="关闭上传"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <p>支持 PDF、Markdown、Word 和 Excel，单个文件不超过 10 MB。</p>
        <label className="drop-zone">
          <Upload size={28} />
          <strong>{file?.name ?? "选择要上传的文件"}</strong>
          <span>原件将保存在当前组织的私有空间</span>
          <input
            aria-label="上传文件"
            type="file"
            accept=".md,.pdf,.docx,.xlsx"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label>
          访问范围
          <select
            aria-label="访问范围"
            value={scope}
            onChange={(e) => setScope(e.target.value)}
          >
            <option value="private">仅自己和管理者</option>
            {user.groups.map((g) => (
              <option key={g} value={g}>
                {groupLabels[g] ?? g}
              </option>
            ))}
            {user.role === "admin" && (
              <option value="public">当前组织所有成员</option>
            )}
          </select>
        </label>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        <div className="modal-actions">
          <button type="button" className="secondary" onClick={onClose}>
            取消
          </button>
          <button className="primary" disabled={!file || busy}>
            {busy ? "正在上传…" : "上传并处理"}
          </button>
        </div>
      </form>
    </div>
  );
}

type Version = {
  version_id: string;
  filename: string;
  media_type: string;
  content_hash: string;
  status: string;
  created_at: string;
  active: boolean;
  chunks: number;
  parser: string;
  chunking: string;
  embedding: string;
};
type Processing = {
  title: string;
  status: string;
  pipeline: Record<string, unknown>;
  original_url: string;
  versions: Version[];
  blocks: { text: string; locator: { label: string } }[];
  chunks: {
    id: string;
    text: string;
    locator: { label: string; token_count_estimate?: number };
  }[];
};
function VersionPanel({ versions }: { versions: Processing["versions"] }) {
  return (
    <section className="version-panel" data-testid="version-panel">
      <h3>
        版本与处理来源
        <span>
          {versions.length === 1
            ? "当前只有一个版本"
            : `共 ${versions.length} 个版本`}
        </span>
      </h3>
      {versions.map((v) => (
        <dl
          className={v.active ? "version active" : "version"}
          key={v.version_id}
          data-version-id={v.version_id}
        >
          <div>
            <dt>状态</dt>
            <dd>
              {v.active ? (
                <b className="tag-yes">当前生效</b>
              ) : (
                <span className="tag-no">历史版本</span>
              )}
              {" · "}
              {statusLabels[v.status] ?? v.status} · {v.chunks} 个分块
            </dd>
          </div>
          <div>
            <dt>原件</dt>
            <dd>
              {v.filename}
              <br />
              <code title="原件内容的 SHA-256">
                {v.content_hash.slice(0, 16)}…
              </code>
            </dd>
          </div>
          <div>
            <dt>解析</dt>
            <dd>{v.parser}</dd>
          </div>
          <div>
            <dt>分块</dt>
            <dd>{v.chunking}</dd>
          </div>
          <div>
            <dt>向量模型</dt>
            <dd>{v.embedding}</dd>
          </div>
          <div>
            <dt>入库时间</dt>
            <dd>{new Date(v.created_at).toLocaleString()}</dd>
          </div>
        </dl>
      ))}
      <small>
        分块的原文位置绑定在上面这个版本上；更换处理配置会产生新版本，不会改写已有分块。
      </small>
    </section>
  );
}

function ProcessingDialog({
  data,
  onClose,
}: {
  data: Processing;
  onClose: () => void;
}) {
  const [view, setView] = useState<"blocks" | "chunks">("blocks");
  return (
    <div className="modal-backdrop">
      <section
        className="processing-modal"
        role="dialog"
        aria-label="解析与分块检查"
      >
        <div className="modal-heading">
          <h2>{data.title}</h2>
          <button
            className="icon-button"
            aria-label="关闭解析检查"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <p>
          处理状态：{statusLabels[data.status] ?? data.status} ·{" "}
          <a href={data.original_url} target="_blank" rel="noreferrer">
            打开原件
          </a>
        </p>
        <div className="processing-tabs">
          <button
            className={view === "blocks" ? "primary" : "secondary"}
            onClick={() => setView("blocks")}
          >
            解析结果（{data.blocks.length}）
          </button>
          <button
            className={view === "chunks" ? "primary" : "secondary"}
            onClick={() => setView("chunks")}
          >
            分块结果（{data.chunks.length}）
          </button>
        </div>
        {data[view].map((item, i) => (
          <article className="processing-block" key={i}>
            <strong>{item.locator.label}</strong>
            <pre>{item.text}</pre>
            <details>
              <summary>查看来源位置</summary>
              <pre>{JSON.stringify(item.locator, null, 2)}</pre>
            </details>
          </article>
        ))}
        <VersionPanel versions={data.versions} />
        <details>
          <summary>原始处理配置</summary>
          <pre>{JSON.stringify(data.pipeline, null, 2)}</pre>
        </details>
      </section>
    </div>
  );
}

function Workspace({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [tab, setTab] = useState<"chat" | "agent" | "documents" | "history">(
    location.hash === "#chat"
      ? "chat"
      : location.hash === "#agent" || user.trial.enabled
        ? "agent"
        : "chat",
  );
  const [docs, setDocs] = useState<Doc[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [history, setHistory] = useState<Result[]>([]);
  const [agentTasks, setAgentTasks] = useState<AgentTask[]>([]);
  const [agentGoal, setAgentGoal] = useState("");
  const [agentMode, setAgentMode] = useState<"auto" | "workflow" | "dynamic">(
    "auto",
  );
  const [agentDocument, setAgentDocument] = useState("");
  const [useMemory, setUseMemory] = useState(false);
  const [memoryText, setMemoryText] = useState("");
  const [memoryBusy, setMemoryBusy] = useState(false);
  const [memoryStatus, setMemoryStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [retryable, setRetryable] = useState(false);
  const [lastQuestion, setLastQuestion] = useState("");
  const [chatRequest, setChatRequest] = useState("");
  const [chatStage, setChatStage] = useState("accepted");
  const [chatStarted, setChatStarted] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [detached, setDetached] = useState(false);
  const [detachReason, setDetachReason] = useState("user");
  const [timings, setTimings] = useState<{
    samples: number;
    median_seconds: number | null;
  } | null>(null);
  const chatAbort = useRef<AbortController | null>(null);
  const delivered = useRef(new Set<string>());
  const [evidenceLoading, setEvidenceLoading] = useState(false);
  const [evidenceError, setEvidenceError] = useState("");
  const lastCitation = useRef<Citation | null>(null);
  const evidenceTrigger = useRef<HTMLButtonElement | null>(null);
  const [evidenceQuotes, setEvidenceQuotes] = useState<string[]>([]);
  const evidenceRequest = useRef(0);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [upload, setUpload] = useState(false);
  const [processing, setProcessing] = useState<Processing | null>(null);
  const [trial, setTrial] = useState<TrialStatus>(user.trial);
  async function preview(id: string) {
    try {
      setProcessing(await api<Processing>("/documents/" + id + "/processing"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const [filter, setFilter] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  async function loadDocs() {
    try {
      setDocs(await api<Doc[]>("/documents"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function loadAgentTasks() {
    try {
      setAgentTasks(await api<AgentTask[]>("/agent/tasks"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function loadTrial() {
    if (!trial.enabled) return;
    try {
      setTrial(
        await api<TrialStatus>("/trial/status", {
          signal: AbortSignal.timeout(8000),
        }),
      );
    } catch {
      /* main action reports errors */
    }
  }
  useEffect(() => {
    loadDocs();
    const timer = setInterval(loadDocs, 5000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (tab === "history")
      api<Result[]>("/history")
        .then(setHistory)
        .catch((e) => setError(e.message));
  }, [tab]);
  useEffect(() => {
    if (tab !== "agent") return;
    loadAgentTasks();
    const timer = setInterval(loadAgentTasks, 1500);
    return () => clearInterval(timer);
  }, [tab]);
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [results, busy]);
  useEffect(() => {
    api<{ samples: number; median_seconds: number | null }>("/chat/timings")
      .then(setTimings)
      .catch(() => {});
  }, [results.length]);
  useEffect(() => {
    if (!chatRequest) return;
    const tick = () =>
      setElapsed(Math.max(0, Math.floor((Date.now() - chatStarted) / 1000)));
    tick();
    const timer = setInterval(tick, 1000);
    const timeout = detached
      ? undefined
      : setTimeout(
          () => {
            setDetachReason("timeout");
            setDetached(true);
            chatAbort.current?.abort();
          },
          Math.max(0, 120000 - (Date.now() - chatStarted)),
        );
    return () => {
      clearInterval(timer);
      clearTimeout(timeout);
    };
  }, [chatRequest, chatStarted, detached]);
  useEffect(() => {
    if (!chatRequest) return;
    let stopped = false,
      polling = false;
    const started = chatStarted;
    const poll = async () => {
      if (polling) return;
      polling = true;
      try {
        const p = await api<{
          stage: string;
          status: string;
          answer_id?: string;
        }>("/chat/progress/" + chatRequest, {
          signal: AbortSignal.timeout(8000),
        });
        if (stopped) return;
        setChatStage(p.stage);
        await loadTrial();
        if (stopped) return;
        if (p.status === "completed" && p.answer_id) {
          const result = await api<Result>("/history/" + p.answer_id, {
            signal: AbortSignal.timeout(8000),
          });
          if (!stopped) {
            if (!delivered.current.has(result.id)) {
              delivered.current.add(result.id);
              setResults((old) => [...old, result]);
            }
            setError("");
            setRetryable(false);
            setChatRequest("");
            setDetached(false);
            setBusy(false);
            setQuestion("");
            await loadTrial();
          }
        } else if (p.status === "failed") {
          setChatRequest("");
          setDetached(false);
          if (detached)
            setError(
              "该请求未完成。已预留的访客额度不退回；请查看剩余额度后再决定是否重试。",
            );
          await loadTrial();
        }
      } catch (e) {
        if (stopped) return;
        const failure = e as Error & { status?: number };
        if (
          detached &&
          ((failure.status === 404 && Date.now() - started > 15000) ||
            Date.now() - started > 900000)
        ) {
          setChatRequest("");
          setDetached(false);
          setError(
            "服务端请求状态暂时无法确认。请先查看问答记录；重试会另计一次额度，已预留额度不退回。",
          );
          await loadTrial();
        }
      } finally {
        polling = false;
      }
    };
    const timer = setInterval(poll, 1200);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [chatRequest, chatStarted, detached]);
  async function cancelAgent(task: AgentTask) {
    try {
      const updated = await api<AgentTask>(`/agent/tasks/${task.id}/cancel`, {
        method: "POST",
        signal: AbortSignal.timeout(15000),
      });
      setAgentTasks((old) =>
        old.map((t) => (t.id === updated.id ? updated : t)),
      );
      await loadTrial();
    } catch (e) {
      setError(
        (e as Error).name === "TimeoutError"
          ? "取消请求尚未确认，请以刷新后的任务状态为准；已预留额度不退回。"
          : (e as Error).message,
      );
    }
  }
  async function ask(e?: React.FormEvent, preset?: string) {
    e?.preventDefault();
    const value = (preset ?? question).trim();
    if (value.length < 2 || busy || chatRequest) return;
    setBusy(true);
    const identifier = crypto.randomUUID();
    const controller = new AbortController();
    chatAbort.current = controller;
    setChatRequest(identifier);
    setChatStarted(Date.now());
    setElapsed(0);
    setChatStage("accepted");
    setDetached(false);
    setError("");
    setRetryable(false);
    setLastQuestion(value);
    setTab("chat");
    setQuestion(value);
    try {
      const result = await api<Result>("/chat", {
        method: "POST",
        signal: controller.signal,
        headers: { "X-Chat-Request": identifier },
        body: JSON.stringify({
          question: value,
          history: results.slice(-10).map((row) => row.question),
        }),
      });
      if (!delivered.current.has(result.id)) {
        delivered.current.add(result.id);
        setResults((old) => [...old, result]);
      }
      setChatRequest("");
      setDetached(false);
      setQuestion("");
      await loadTrial();
    } catch (e) {
      if (controller.signal.aborted) return;
      const gateway = e as Error & { status?: number; stage?: string };
      if (
        e instanceof TypeError ||
        (gateway.status &&
          [502, 503, 504].includes(gateway.status) &&
          !gateway.stage)
      ) {
        setDetachReason("connection");
        setDetached(true);
        setError(
          "连接中断，服务端结果尚待确认。页面会继续查询状态；不要重复提交。已预留的访客额度不退回。",
        );
        return;
      }
      setChatRequest("");
      const failure = e as Error & {
        retryable?: boolean;
        stage?: string;
        modelReached?: boolean;
      };
      // Only say the model was never reached when the server actually knows that.
      // Once generation has started, the outcome of this request is unknown, and
      // claiming otherwise would tell the reader that retrying is free when it is not.
      setError(
        failure.retryable
          ? `${failure.message}${outageNote(failure)}${trial.enabled ? "额度以顶部剩余次数为准；已预留额度不退回，重新提交会计入新请求。" : ""}`
          : failure.message,
      );
      setRetryable(Boolean(failure.retryable));
      await loadTrial();
    } finally {
      setBusy(false);
    }
  }
  function closeEvidence() {
    evidenceRequest.current++;
    setEvidenceLoading(false);
    setEvidenceError("");
    lastCitation.current = null;
    setEvidence(null);
    if (evidenceTrigger.current?.isConnected)
      evidenceTrigger.current.focus({ preventScroll: true });
  }
  async function openEvidence(c: Citation, quotes: string[]) {
    const active = document.activeElement;
    if (
      active instanceof HTMLButtonElement &&
      !active.closest(".evidence-pane")
    )
      evidenceTrigger.current = active;
    const identifier = ++evidenceRequest.current;
    lastCitation.current = c;
    setEvidenceLoading(true);
    setEvidenceError("");
    setEvidenceQuotes(quotes);
    try {
      const source = await api<Evidence>("/evidence/" + c.chunk_id, {
        signal: AbortSignal.timeout(15000),
      });
      if (identifier === evidenceRequest.current) setEvidence(source);
    } catch (e) {
      if (identifier === evidenceRequest.current) {
        setEvidence(null);
        setEvidenceError(
          (e as Error).name === "TimeoutError"
            ? "引用加载超时，请重试。"
            : (e as Error).message,
        );
      }
    } finally {
      if (identifier === evidenceRequest.current) setEvidenceLoading(false);
    }
  }
  async function startAgent(e: React.FormEvent) {
    e.preventDefault();
    if (
      agentGoal.trim().length < 4 ||
      busy ||
      agentTasks.some((t) => ["queued", "running"].includes(t.status))
    )
      return;
    setBusy(true);
    setError("");
    try {
      const task = await api<AgentTask>("/agent/tasks", {
        method: "POST",
        body: JSON.stringify({
          goal: agentGoal.trim(),
          mode: agentMode,
          max_steps: agentMode === "dynamic" ? 6 : 4,
          document_id: agentDocument || null,
          use_memory: useMemory,
        }),
      });
      setAgentTasks((old) => [
        task,
        ...old.filter((row) => row.id !== task.id),
      ]);
      setAgentGoal("");
      await loadTrial();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function saveMemory(e: React.FormEvent) {
    e.preventDefault();
    const content = memoryText.trim();
    if (content.length < 4 || memoryBusy) return;
    setMemoryBusy(true);
    setMemoryStatus("");
    try {
      await api("/agent/memories", {
        method: "POST",
        body: JSON.stringify({ content }),
      });
      setMemoryText("");
      setMemoryStatus(
        "已提交到你的长期记忆；仅在任务中开启“使用长期记忆”时检索。",
      );
    } catch (e) {
      setMemoryStatus((e as Error).message);
    } finally {
      setMemoryBusy(false);
    }
  }
  async function retry(doc: Doc) {
    try {
      await api("/documents/" + doc.id + "/retry", { method: "POST" });
      await loadDocs();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const ready = docs.filter((d) => d.status === "ready").length;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <BookOpen size={22} />
          <span>星桥知识库</span>
        </div>
        <div className="workspace-label">
          {user.tenant}
          <span>工作空间</span>
        </div>
        <nav>
          <button
            aria-label="知识 Agent"
            className={tab === "agent" ? "active" : ""}
            onClick={() => setTab("agent")}
          >
            <Workflow size={18} />
            知识 Agent
          </button>
          <button
            aria-label="知识问答"
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            <Search size={18} />
            知识问答
          </button>
          <button
            aria-label="资料库"
            className={tab === "documents" ? "active" : ""}
            onClick={() => setTab("documents")}
          >
            <FolderOpen size={18} />
            资料库<span>{docs.length}</span>
          </button>
          <button
            aria-label="问答记录"
            className={tab === "history" ? "active" : ""}
            onClick={() => setTab("history")}
          >
            <History size={18} />
            问答记录
          </button>
        </nav>
        <div className="sidebar-section-label">当前知识范围</div>
        <div className="scope-card">
          <ShieldCheck size={17} />
          <div>
            <strong>
              {user.role === "admin"
                ? "组织管理者"
                : user.groups.map((g) => groupLabels[g] ?? g).join("、")}
            </strong>
            <p>仅检索你有权查看的资料</p>
          </div>
        </div>
        <div className="sidebar-spacer" />
        <div className="local-label">
          <span />
          本地模型 · API 费用 0
        </div>
        <div className="profile">
          <div className="avatar">{user.name[0]}</div>
          <div>
            <strong>{user.name}</strong>
            <small>{user.username}</small>
          </div>
          {!publicDemo && (
            <button
              title="退出登录"
              aria-label="退出登录"
              className="icon-button"
              onClick={async () => {
                await api("/auth/logout", { method: "POST" });
                onLogout();
              }}
            >
              <LogOut size={17} />
            </button>
          )}
        </div>
      </aside>
      <main className="main-area">
        <header className="topbar">
          <div>
            <span className="breadcrumb">工作空间</span>
            <ChevronRight size={13} />
            <strong>
              {tab === "chat"
                ? "知识问答"
                : tab === "agent"
                  ? "知识 Agent"
                  : tab === "documents"
                    ? "资料库"
                    : "问答记录"}
            </strong>
          </div>
          <div className="topbar-right">
            <a href="/" className="project-link">
              Agentic-RAG · 项目介绍
            </a>
            <a
              href="https://github.com/Mingjie-Mao/Agentic-RAG"
              className="project-link"
            >
              GitHub ↗
            </a>
            {trial.enabled ? (
              <span className="stage-badge">
                演示访客 · 共享剩余 {trial.remaining ?? 0}/{trial.limit ?? 0}
              </span>
            ) : (
              <button className="secondary" onClick={() => setUpload(true)}>
                <Upload size={15} />
                添加资料
              </button>
            )}
          </div>
        </header>
        {publicDemo && (
          <p className="public-demo-notice">
            虚构企业资料 · 可体验问答、Agent 查证与原文核验 · 每日共享 {trial.limit ?? 10} 次问答 /
            Agent 额度，UTC 零点重置。
          </p>
        )}
        {trial.enabled && (trial.remaining ?? 0) <= 0 && (
          <div className="quota-note" role="status">
            今日共享额度已用完，UTC 零点重置。仍可查阅资料与历史记录，或
            <a href="/replay.html">查看不消耗额度的真实回放</a>。
          </div>
        )}
        {error && (
          <div className="error-banner" role="alert" data-testid="error-banner">
            <span className="retry-line">
              {error}
              {retryable && (
                <button
                  data-testid="retry-question"
                  disabled={
                    busy ||
                    Boolean(chatRequest) ||
                    (trial.enabled && (trial.remaining ?? 0) <= 0)
                  }
                  onClick={() => ask(undefined, lastQuestion)}
                >
                  重试这个问题
                </button>
              )}
            </span>
            <button
              className="icon-button"
              onClick={() => setError("")}
              aria-label="关闭提示"
            >
              <X size={15} />
            </button>
          </div>
        )}
        {tab === "agent" ? (
          <div className="conversation-layout">
            <section className="agent-page">
              <div className="agent-heading">
                <span className="eyebrow">ENTERPRISE KNOWLEDGE AGENT</span>
                <h1>让 Agent 分步查证资料</h1>
                <p>
                  比较政策变化、综合多份资料或核对矛盾原文。资料读取每一步重新鉴权，结果附可查看的引用。
                </p>
                <p className="demo-scope">
                  {trial.enabled
                    ? "当前访客演示使用固定工作流，自动路由也执行该路径。动态模式和长期记忆仅在本地实验环境开放。"
                    : "默认自动路由；动态模式与长期记忆为实验功能。"}
                </p>
                <details className="experiment-details">
                  <summary>执行方式与实验边界</summary>
                  <p>
                    LangGraph
                    负责节点调度和检查点；授权工具、累计预算与输出核验由业务层执行。动态模式的质量优势尚未得到正式
                    Core 验收证明。
                  </p>
                </details>
                <a
                  className="role-comparison-link"
                  href="/replay.html#permissions"
                  target="_blank"
                  rel="noreferrer"
                >
                  查看固定虚构角色的权限对照（历史快照） ↗
                </a>
                <div className="agent-examples">
                  {agentExamples.map((example) => (
                    <button
                      key={example.title}
                      onClick={() => {
                        setAgentGoal(example.goal);
                        setAgentDocument("");
                      }}
                      disabled={busy}
                    >
                      <strong>{example.title}</strong>
                      <span>{example.detail}</span>
                    </button>
                  ))}
                </div>
              </div>
              {!trial.enabled && (
                <form className="memory-composer" onSubmit={saveMemory}>
                  <div>
                    <strong>长期记忆</strong>
                    <small>
                      显式写入，会调用已配置的记忆抽取模型；不会把 Agent
                      回答自动永久保存。
                    </small>
                  </div>
                  <div className="memory-row">
                    <input
                      aria-label="写入长期记忆"
                      placeholder="例如：我负责支持团队，回答时优先给出简短检查清单"
                      value={memoryText}
                      onChange={(e) => setMemoryText(e.target.value)}
                      maxLength={2000}
                    />
                    <button
                      className="secondary"
                      disabled={memoryBusy || memoryText.trim().length < 4}
                    >
                      {memoryBusy ? "写入中…" : "记住"}
                    </button>
                  </div>
                  {memoryStatus && (
                    <p className="memory-status" role="status">
                      {memoryStatus}
                    </p>
                  )}
                </form>
              )}
              <form className="agent-composer" onSubmit={startAgent}>
                <textarea
                  aria-label="Agent 任务目标"
                  placeholder="例如：对照 2025 与 2026 差旅政策，说明悉尼住宿标准发生什么变化并引用原文"
                  value={agentGoal}
                  onChange={(e) => setAgentGoal(e.target.value)}
                  maxLength={1500}
                  rows={3}
                />
                <div className="agent-options">
                  <label>
                    执行方式
                    <select
                      value={agentMode}
                      onChange={(e) =>
                        setAgentMode(
                          e.target.value as "auto" | "workflow" | "dynamic",
                        )
                      }
                    >
                      <option value="auto">自动路由（推荐）</option>
                      <option value="workflow">固定工作流</option>
                      {!trial.enabled && (
                        <option value="dynamic">动态 Agent（实验）</option>
                      )}
                    </select>
                  </label>
                  <label>
                    指定资料（可选）
                    <select
                      value={agentDocument}
                      onChange={(e) => setAgentDocument(e.target.value)}
                    >
                      <option value="">跨资料检索</option>
                      {docs
                        .filter((doc) => doc.status === "ready")
                        .map((doc) => (
                          <option key={doc.id} value={doc.id}>
                            {doc.title}
                          </option>
                        ))}
                    </select>
                  </label>
                  {!trial.enabled && (
                    <label>
                      <span>
                        <input
                          type="checkbox"
                          checked={useMemory}
                          onChange={(e) => setUseMemory(e.target.checked)}
                        />{" "}
                        使用长期记忆（增加延迟与 token）
                      </span>
                    </label>
                  )}
                  <button
                    className="primary"
                    disabled={
                      busy ||
                      (trial.enabled && (trial.remaining ?? 0) <= 0) ||
                      agentGoal.trim().length < 4 ||
                      agentTasks.some((t) =>
                        ["queued", "running"].includes(t.status),
                      )
                    }
                  >
                    {busy ? "正在提交任务…" : "开始任务"}
                  </button>
                </div>
              </form>
              <div className="agent-task-list">
                {agentTasks.length === 0 ? (
                  <p className="muted">
                    选择上方示例，点击“开始任务”，查看实际查证步骤与引用。
                  </p>
                ) : (
                  agentTasks.map((task) => (
                    <article className="agent-task" key={task.id}>
                      <header>
                        <div>
                          <span className="status-pill ready">
                            {(
                              {
                                queued: "排队中",
                                running: "执行中",
                                completed: "执行结束",
                                failed: "执行失败",
                                cancelled: "已取消",
                              } as Record<string, string>
                            )[task.status] ?? "状态待确认"}
                          </span>
                          <strong>{task.goal}</strong>
                        </div>
                        <small>
                          {modeLabels[task.mode] ?? "执行方式待确认"} ·{" "}
                          {task.step_no}/{task.max_steps} 步
                        </small>
                      </header>
                      {task.result?.status && (
                        <AnswerCard
                          result={{
                            ...task.result,
                            id: task.id,
                            question: task.goal,
                          }}
                          onEvidence={openEvidence}
                          showQuestion={false}
                        />
                      )}
                      {task.error && (
                        <details className="doc-error">
                          <summary>任务未完成 · 查看错误详情</summary>
                          <p>{task.error}</p>
                        </details>
                      )}
                      <AgentTimeline
                        task={task}
                        onCancel={() => cancelAgent(task)}
                      />
                    </article>
                  ))
                )}
              </div>
            </section>
            <EvidencePane
              evidence={evidence}
              loading={evidenceLoading}
              quotes={evidenceQuotes}
              error={evidenceError}
              onRetry={() => {
                if (lastCitation.current)
                  openEvidence(lastCitation.current, evidenceQuotes);
              }}
              onClose={closeEvidence}
            />
          </div>
        ) : tab === "documents" ? (
          <section className="documents-page">
            <div className="page-title">
              <div>
                <span className="eyebrow">DOCUMENTS</span>
                <h1>你的资料库</h1>
                <p>{ready} 份资料可检索，所有原文都保留在当前组织。</p>
              </div>
              <div className="search-box">
                <Search size={17} />
                <input
                  aria-label="搜索文档"
                  placeholder="按标题查找资料"
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                />
              </div>
            </div>
            <div className="document-table">
              <div className="doc-table-head">
                <span>资料名称</span>
                <span>访问范围</span>
                <span>处理状态</span>
              </div>
              {docs
                .filter((d) => d.title.includes(filter))
                .map((doc) => (
                  <div
                    className="doc-row"
                    key={doc.id}
                    data-document-id={doc.id}
                  >
                    <div className="doc-name">
                      <div className="file-icon">
                        <FileText size={20} />
                      </div>
                      <div>
                        <button
                          className="document-title"
                          onClick={() => preview(doc.id)}
                        >
                          {doc.title}
                        </button>
                        <small>
                          {doc.filename.split(".").pop()?.toUpperCase() ??
                            "文档"}{" "}
                          · {doc.chunks} 个片段
                        </small>
                        {doc.error && <p className="doc-error">{doc.error}</p>}
                      </div>
                    </div>
                    <span className="doc-scope">
                      {doc.tenant_public
                        ? "组织共享"
                        : doc.groups
                            .map((g) => groupLabels[g] ?? g)
                            .join("、") || "私有"}
                    </span>
                    <div>
                      <span className={"status-pill " + doc.status}>
                        {doc.status === "processing" ? (
                          <LoaderCircle className="spin" size={12} />
                        ) : doc.status === "ready" ? (
                          <Check size={12} />
                        ) : (
                          <Clock3 size={12} />
                        )}{" "}
                        {statusLabels[doc.status]}
                      </span>
                      {doc.status === "failed" && (
                        <button className="retry" onClick={() => retry(doc)}>
                          重试
                        </button>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </section>
        ) : (
          <div className="conversation-layout">
            <section className="conversation-column">
              <div className="conversation-scroll">
                {tab === "chat" && (
                  <details
                    className="experience-guide"
                    open={results.length === 0}
                  >
                    <summary>选择体验场景</summary>
                    <p className="muted">
                      点击只填入示例问题，不消耗额度；点击发送后开始查证。
                    </p>
                    <div className="suggestions">
                      {chatExamples.map((example, i) => (
                        <button
                          key={example.title}
                          disabled={busy || Boolean(chatRequest)}
                          onClick={() => setQuestion(example.question)}
                        >
                          <span className="suggestion-number">0{i + 1}</span>
                          <span>
                            <strong>{example.title}</strong>
                            <small>{example.detail}</small>
                          </span>
                          <ChevronRight size={16} />
                        </button>
                      ))}
                    </div>
                    <a
                      href="/replay.html#permissions"
                      target="_blank"
                      rel="noreferrer"
                    >
                      查看固定虚构角色的权限对照（历史快照） ↗
                    </a>
                  </details>
                )}
                {tab === "history" ? (
                  <>
                    <div className="history-heading">
                      <span className="eyebrow">HISTORY</span>
                      <h1>问答记录</h1>
                      <p>
                        只有你能访问自己的记录；引用权限变化后会隐藏相关内容。
                      </p>
                    </div>
                    {history.length ? (
                      history.map((r) => (
                        <AnswerCard
                          key={r.id}
                          result={r}
                          onEvidence={openEvidence}
                        />
                      ))
                    ) : (
                      <p className="muted">还没有问答记录。</p>
                    )}
                  </>
                ) : results.length ? (
                  <>
                    {results.map((r) => (
                      <AnswerCard
                        key={r.id}
                        result={r}
                        onEvidence={openEvidence}
                      />
                    ))}
                  </>
                ) : (
                  <div className="welcome">
                    <span className="eyebrow">连接资料与答案</span>
                    <h1>有什么想了解的？</h1>
                    <p>
                      从 {ready} 份可访问资料中查找依据。
                      <br />
                      查看原文，再做判断。
                    </p>
                    <div className="trust-note">
                      <ShieldCheck size={15} />
                      你的访问权限也会用于检索和引用
                    </div>
                  </div>
                )}
                {chatRequest && (
                  <div className="thinking" role="status">
                    <LoaderCircle className="spin" size={18} />
                    <div>
                      {(
                        {
                          accepted: "请求已提交，等待服务端确认",
                          retrieval: "按权限检索资料",
                          generation: "正在生成有依据的答案",
                          verification: "正在核对引用与事实，必要时修复",
                          authorization: "返回前再次检查资料权限",
                          completed: "已完成",
                        } as Record<string, string>
                      )[chatStage] ?? "等待服务端状态"}
                      <span>
                        已等待 {elapsed} 秒 ·{" "}
                        {timings?.samples
                          ? `最近 ${timings.samples} 次可访问问答的中位耗时 ${timings.median_seconds} 秒`
                          : "尚无近期耗时样本，本地模型可能需要几十秒"}
                      </span>
                      {detached ? (
                        <span>
                          {detachReason === "timeout"
                            ? "原请求已等待 120 秒，客户端停止等待。"
                            : detachReason === "connection"
                              ? "连接已中断。"
                              : "已停止等待原请求。"}
                          服务端仍可能执行；正在查询结果。已预留额度不退回。
                        </span>
                      ) : (
                        <button
                          className="secondary"
                          onClick={() => {
                            setDetachReason("user");
                            setDetached(true);
                            chatAbort.current?.abort();
                          }}
                        >
                          停止等待
                        </button>
                      )}
                      {elapsed >= 120 && (
                        <span>
                          等待较久，当前结果尚未确认。可打开问答记录查看已完成结果；重试会另计一次额度。
                        </span>
                      )}
                    </div>
                  </div>
                )}
                <div ref={bottom} />
              </div>
              {tab === "chat" && (
                <div className="composer-wrap">
                  <form className="composer" onSubmit={(e) => ask(e)}>
                    <textarea
                      aria-label="输入问题"
                      placeholder="向你的企业资料提问…"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      maxLength={1000}
                      rows={2}
                      onKeyDown={(e) => {
                        if (
                          e.key === "Enter" &&
                          !e.shiftKey &&
                          !e.nativeEvent.isComposing
                        ) {
                          e.preventDefault();
                          ask();
                        }
                      }}
                    />
                    <div className="composer-bottom">
                      <span>
                        <BookOpen size={14} />
                        当前组织 · 按权限检索
                      </span>
                      <button
                        className="new-question"
                        type="button"
                        title="开始独立问题，不关联上文；已保存的问答记录仍可查看"
                        disabled={
                          busy || Boolean(chatRequest) || !results.length
                        }
                        onClick={() => {
                          setResults([]);
                          setQuestion("");
                        }}
                      >
                        新问题
                      </button>
                      <button
                        className="send-button"
                        aria-label="发送问题"
                        disabled={
                          busy ||
                          Boolean(chatRequest) ||
                          (trial.enabled && (trial.remaining ?? 0) <= 0) ||
                          question.trim().length < 2
                        }
                      >
                        {busy ? (
                          <LoaderCircle className="spin" size={19} />
                        ) : (
                          <ArrowUp size={19} />
                        )}
                      </button>
                    </div>
                  </form>
                  <p className="composer-note">
                    {timings?.samples
                      ? `最近 ${timings.samples} 次可访问问答的中位耗时 ${timings.median_seconds} 秒。`
                      : "近期耗时尚未测得。"}
                  </p>
                  <p className="composer-note">
                    回答由模型生成，请结合引用资料核验。Shift + Enter 换行。
                  </p>
                </div>
              )}
            </section>
            <EvidencePane
              evidence={evidence}
              loading={evidenceLoading}
              quotes={evidenceQuotes}
              error={evidenceError}
              onRetry={() => {
                if (lastCitation.current)
                  openEvidence(lastCitation.current, evidenceQuotes);
              }}
              onClose={closeEvidence}
            />
          </div>
        )}
      </main>
      {processing && (
        <ProcessingDialog
          data={processing}
          onClose={() => setProcessing(null)}
        />
      )}
      {upload && (
        <UploadDialog
          user={user}
          onClose={() => setUpload(false)}
          onDone={loadDocs}
        />
      )}
    </div>
  );
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState(true);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (publicDemo) {
      let stopped = false,
        pending = false,
        connected = false;
      const connect = async () => {
        if (pending || connected || stopped) return;
        pending = true;
        setConnecting(true);
        try {
          const visitor = await connectPublicSession(
            () => api<User>("/auth/me", { signal: AbortSignal.timeout(3000) }),
            renewVisitorSession,
          );
          if (!stopped) {
            connected = true;
            setUser(visitor);
          }
        } catch {
          // The static replay remains usable while the live backend is unavailable.
        } finally {
          pending = false;
          if (!stopped) setConnecting(false);
        }
      };
      void connect();
      const timer = window.setInterval(connect, 30000);
      window.addEventListener("online", connect);
      return () => {
        stopped = true;
        window.clearInterval(timer);
        window.removeEventListener("online", connect);
      };
    }
    api<User>("/auth/me", { signal: AbortSignal.timeout(8000) })
      .then(setUser)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [attempt]);
  if (publicDemo)
    return user ? (
      <Workspace key={user.id} user={user} onLogout={() => setUser(null)} />
    ) : (
      <PublicDemoFallback
        connecting={connecting}
        onRetry={() => setAttempt((n) => n + 1)}
      />
    );
  if (loading)
    return (
      <div className="loading-screen">
        <LoaderCircle className="spin" />
        正在打开工作空间
      </div>
    );
  return user ? (
    <Workspace key={user.id} user={user} onLogout={() => setUser(null)} />
  ) : (
    <Login onLogin={setUser} />
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
