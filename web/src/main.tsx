import React, { useEffect, useRef, useState } from "react";
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
  X,
  ExternalLink,
  PanelRightClose,
} from "lucide-react";
import "./style.css";

type User = {
  id: string;
  name: string;
  username: string;
  tenant: string;
  tenant_id: string;
  role: string;
  groups: string[];
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
  locator: { label: string; page?: number };
  text: string;
  original_url: string;
  preview_url: string;
};
type Result = {
  id: string;
  question: string;
  status: string;
  claims: { text: string; evidence_ids: string[]; quotes: string[] }[];
  citations: Citation[];
  message: string;
  trace?: {
    method: string;
    embedding_model: string;
    generation_model: string;
    candidates: { title: string; score: number }[];
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

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch("/api" + path, {
    ...options,
    credentials: "same-origin",
    headers: {
      "X-Requested-With": "EnterpriseRAG",
      ...(options.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...options.headers,
    },
  });
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: "服务暂时不可用" }));
    const detail =
      typeof body.detail === "string" ? body.detail : "请求内容不符合要求";
    throw new Error(detail);
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
const prompts = [
  "星桥标准套餐每分钟可以调用多少次 API？",
  "2026 年悉尼出差的酒店每晚报销上限是多少澳元？",
  "错误 E401 表示什么，应该怎么处理？",
];

function Login({ onLogin }: { onLogin: (u: User) => void }) {
  const [accounts, setAccounts] = useState<User[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
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
  return (
    <main className="login-shell">
      <section className="login-story">
        <div className="brand">
          <BookOpen size={23} />
          <span>星桥知识库</span>
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
          <h2>进入你的工作空间</h2>
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
        </form>
      </section>
    </main>
  );
}

function EvidencePane({
  evidence,
  onClose,
}: {
  evidence: Evidence | null;
  onClose: () => void;
}) {
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
    <aside className="evidence-pane">
      <div className="evidence-heading">
        <span>来源与证据</span>
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
        <div className="source-text">{evidence.text}</div>
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
}: {
  result: Result;
  onEvidence: (c: Citation) => void;
}) {
  const [trace, setTrace] = useState(false);
  return (
    <article className="answer-card" data-testid="answer-card">
      <div className="answer-question">{result.question}</div>
      <div className="answer-label">
        <span className="answer-logo">
          <BookOpen size={17} />
        </span>
        知识助手
        <span className="answer-status">
          {["answered", "conflict"].includes(result.status)
            ? "附原文依据"
            : result.status === "access_changed"
              ? "权限已变化"
              : "依据不足"}
        </span>
      </div>
      {result.status === "conflict" && <p className="conflict-notice">{result.message}</p>}
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
                      onClick={() => onEvidence(result.citations[i])}
                    >
                      {i + 1}
                    </button>
                  ) : null;
                })}
              </span>
            </p>
          ))}
        </div>
      ) : (
        <p className="no-answer">{result.message}</p>
      )}
      {result.citations.length > 0 && (
        <div className="citation-grid">
          {result.citations.map((c, index) => (
            <button
              className="citation-card"
              key={c.chunk_id}
              onClick={() => onEvidence(c)}
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
            <div className="trace-panel">
              <div className="trace-stats">
                <span>
                  向量检索 <b>{result.trace.retrieval_ms.toFixed(0)} ms</b>
                </span>
                <span>
                  生成 <b>{(result.trace.generation_ms / 1000).toFixed(1)} s</b>
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
                {result.trace.embedding_model} → {result.trace.generation_model}
              </p>
              <ol>
                {result.trace.candidates.map((c, i) => (
                  <li key={i}>
                    <span>{c.title}</span>
                    <code>{c.score.toFixed(3)}</code>
                  </li>
                ))}
              </ol>
              <small>分数表示向量相似度，不代表答案正确概率。</small>
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

type Processing = {title:string;status:string;pipeline:Record<string,unknown>;original_url:string;blocks:{text:string;locator:{label:string}}[];chunks:{id:string;text:string;locator:{label:string;token_count_estimate?:number}}[]};
function ProcessingDialog({data,onClose}:{data:Processing;onClose:()=>void}) {
  const [view,setView]=useState<'blocks'|'chunks'>('blocks');
  return <div className="modal-backdrop"><section className="processing-modal" role="dialog" aria-label="解析与分块检查"><div className="modal-heading"><h2>{data.title}</h2><button className="icon-button" aria-label="关闭解析检查" onClick={onClose}><X size={20}/></button></div><p>处理状态：{statusLabels[data.status]??data.status} · <a href={data.original_url} target="_blank" rel="noreferrer">打开原件</a></p><div className="processing-tabs"><button className={view==='blocks'?'primary':'secondary'} onClick={()=>setView('blocks')}>解析结果（{data.blocks.length}）</button><button className={view==='chunks'?'primary':'secondary'} onClick={()=>setView('chunks')}>分块结果（{data.chunks.length}）</button></div>{data[view].map((item,i)=><article className="processing-block" key={i}><strong>{item.locator.label}</strong><pre>{item.text}</pre><details><summary>查看来源位置</summary><pre>{JSON.stringify(item.locator,null,2)}</pre></details></article>)}<details><summary>处理配置</summary><pre>{JSON.stringify(data.pipeline,null,2)}</pre></details></section></div>;
}

function Workspace({ user, onLogout }: { user: User; onLogout: () => void }) {
  const [tab, setTab] = useState<"chat" | "documents" | "history">("chat");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [history, setHistory] = useState<Result[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [upload, setUpload] = useState(false);
  const [processing,setProcessing]=useState<Processing|null>(null);
  async function preview(id:string){try{setProcessing(await api<Processing>("/documents/"+id+"/processing"));}catch(e){setError((e as Error).message);}}
  const [filter, setFilter] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  async function loadDocs() {
    try {
      setDocs(await api<Doc[]>("/documents"));
    } catch (e) {
      setError((e as Error).message);
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
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [results, busy]);
  async function ask(e?: React.FormEvent, preset?: string) {
    e?.preventDefault();
    const value = (preset ?? question).trim();
    if (value.length < 2 || busy) return;
    setBusy(true);
    setError("");
    setTab("chat");
    setQuestion(value);
    try {
      const result = await api<Result>("/chat", {
        method: "POST",
        body: JSON.stringify({ question: value }),
      });
      setResults((old) => [...old, result]);
      setQuestion("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function openEvidence(c: Citation) {
    try {
      setEvidence(await api<Evidence>("/evidence/" + c.chunk_id));
    } catch (e) {
      setEvidence(null);
      setError((e as Error).message);
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
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            <Search size={18} />
            知识问答
          </button>
          <button
            className={tab === "documents" ? "active" : ""}
            onClick={() => setTab("documents")}
          >
            <FolderOpen size={18} />
            资料库<span>{docs.length}</span>
          </button>
          <button
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
                : tab === "documents"
                  ? "资料库"
                  : "问答记录"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="stage-badge">S2/S3 · 开发版</span>
            <button className="secondary" onClick={() => setUpload(true)}>
              <Upload size={15} />
              添加资料
            </button>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            {error}
            <button
              className="icon-button"
              onClick={() => setError("")}
              aria-label="关闭提示"
            >
              <X size={15} />
            </button>
          </div>
        )}
        {tab === "documents" ? (
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
                  <div className="doc-row" key={doc.id} data-document-id={doc.id}>
                    <div className="doc-name">
                      <div className="file-icon">
                        <FileText size={20} />
                      </div>
                      <div>
                        <button className="document-title" onClick={()=>preview(doc.id)}>{doc.title}</button>
                        <small>
                          {doc.filename.split(".").pop()?.toUpperCase() ?? "文档"} ·{" "}
                          {doc.chunks} 个片段
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
                    <div className="suggestions">
                      {prompts.map((q, i) => (
                        <button
                          key={q}
                          onClick={() => ask(undefined, q)}
                          disabled={busy}
                        >
                          <span className="suggestion-number">0{i + 1}</span>
                          <span>{q}</span>
                          <ChevronRight size={16} />
                        </button>
                      ))}
                    </div>
                    <div className="trust-note">
                      <ShieldCheck size={15} />
                      你的访问权限也会用于检索和引用
                    </div>
                  </div>
                )}
                {busy && (
                  <div className="thinking" role="status">
                    <LoaderCircle className="spin" size={18} />
                    <div>
                      正在查阅资料并核对引用
                      <span>本地模型正在生成，请稍候</span>
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
                        className="send-button"
                        aria-label="发送问题"
                        disabled={busy || question.trim().length < 2}
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
                    回答由模型生成，请结合引用资料核验。Shift + Enter 换行。
                  </p>
                </div>
              )}
            </section>
            <EvidencePane
              evidence={evidence}
              onClose={() => setEvidence(null)}
            />
          </div>
        )}
      </main>
      {processing && <ProcessingDialog data={processing} onClose={()=>setProcessing(null)}/>}
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
  useEffect(() => {
    api<User>("/auth/me")
      .then(setUser)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);
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
