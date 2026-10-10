import React, { useEffect, useState } from "react";
import { Clock3, Check, CircleAlert, LoaderCircle } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AgentTask, AgentEvent } from "./main";

export const agentExamples = [
  {
    title: "比较政策变化",
    detail: "对照两份资料，列出金额与适用年份",
    goal: "对照差旅标准 2025 旧版与差旅标准 2026 生效版，悉尼酒店每晚报销上限分别是多少？",
  },
  {
    title: "跨资料查证",
    detail: "结合配额说明与错误处理，给出检查步骤",
    goal: "星桥标准套餐每分钟允许多少次 API 请求？超过配额会返回什么状态码，应该如何处理？",
  },
  {
    title: "核对冲突依据",
    detail: "读取两份试点安排，确认能否得出一致结论",
    goal: "试点项目支持安排 A 和 B 对 2026 年 9 月的支持时间分别如何规定？是否存在冲突？",
  },
];
export const chatExamples = [
  {
    title: "查答案并核对引用",
    detail: "查看答案，再点击引用核对支持它的原句",
    question: "星桥标准套餐每分钟可以调用多少次 API？",
  },
  {
    title: "比较政策版本",
    detail: "对照两个年份的政策，检查金额与适用年份",
    question: "2025 年和 2026 年悉尼酒店每晚报销上限分别是多少澳元？",
  },
  {
    title: "查看资料冲突",
    detail: "检查两份资料的矛盾原文，而非只看模型结论",
    question: "2026 年 9 月试点项目的支持时间是什么？请核对支持安排 A 和 B。",
  },
  {
    title: "体验依据不足时拒答",
    detail: "观察资料没有依据时是否明确拒答",
    question: "公司去火星出差的补贴是多少？",
  },
];
export const modeLabels: Record<string, string> = {
  auto: "自动路由",
  workflow: "固定工作流",
  hybrid: "有界 Hybrid",
  planner: "证据规划",
  dynamic: "动态 Agent",
  adaptive: "自适应策略",
};
const tools: Record<string, string> = {
  search_documents: "查找可访问资料",
  retrieve_evidence: "提取原文证据",
  open_document: "阅读指定资料",
  get_document_version: "查看文档版本",
  compare_versions: "比较版本差异",
  verify_chunk_access: "重新核对访问权限",
  search_memory: "读取已授权偏好",
};
const events: Record<string, string> = {
  task_created: "任务已创建",
  task_queued: "任务已排队",
  task_started: "开始查证",
  tool_completed: "完成资料查阅",
  tool_rejected: "工具调用未执行",
  task_completed: "返回核验结果",
  task_failed: "任务执行失败",
  task_cancelled: "任务已取消",
  budget_exhausted: "已达到执行预算",
  workflow_stop: "工作流结束",
  generation_started: "生成答案",
  generation_context: "准备生成所需证据",
  coverage_check: "检查问题覆盖情况",
  version_plan: "准备版本对照",
  version_route: "确定版本范围",
  effective_selection: "确定适用资料",
  checkpoint_resumed: "从检查点恢复",
  policy_decision: "决定下一步查证",
  evidence_validation: "核对证据",
};
export function eventLabel(event: AgentEvent) {
  return event.tool_name
    ? (event.event_type === "tool_rejected" ? "未执行：" : "") +
        (tools[event.tool_name] ?? "调用授权工具")
    : (events[event.event_type] ?? "更新查证状态");
}
export function elapsedAt(created: string, ended?: string) {
  const parse = (v: string) =>
    Date.parse(/(Z|[+-]\d\d:\d\d)$/.test(v) ? v : v + "Z");
  return Math.max(
    0,
    Math.round(((ended ? parse(ended) : Date.now()) - parse(created)) / 1000),
  );
}
export function AgentTimeline({
  task,
  onCancel,
}: {
  task: AgentTask;
  onCancel: () => Promise<void>;
}) {
  const running = ["queued", "running"].includes(task.status);
  const [cancelling, setCancelling] = useState(false);
  const [, tick] = useState(0);
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => tick((v) => v + 1), 1000);
    return () => clearInterval(timer);
  }, [running]);
  return (
    <div className="task-progress">
      <div className="task-progress-heading" role="status">
        {running ? (
          <LoaderCircle size={16} className="spin" />
        ) : task.status === "completed" ? (
          <Check size={16} />
        ) : (
          <CircleAlert size={16} />
        )}
        <strong>
          {(
            {
              queued: "等待执行",
              running: "正在查证资料",
              completed: "任务已完成",
              failed: "任务未完成",
              cancelled: "任务已取消",
            } as Record<string, string>
          )[task.status] ?? "任务状态待确认"}
        </strong>
        <span>
          <Clock3 size={14} />{" "}
          {elapsedAt(task.created_at, running ? undefined : task.updated_at)} 秒
        </span>
        {running && (
          <button
            className="secondary"
            disabled={cancelling}
            onClick={async () => {
              setCancelling(true);
              try {
                await onCancel();
              } finally {
                setCancelling(false);
              }
            }}
          >
            {cancelling ? "正在提交取消…" : "取消任务"}
          </button>
        )}
      </div>
      <ol className="task-timeline" aria-label="实际执行步骤">
        {task.events
          .filter(
            (event) =>
              event.tool_name ||
              [
                "task_created",
                "task_queued",
                "task_started",
                "generation_started",
                "generation_context",
                "coverage_check",
                "version_plan",
                "version_route",
                "task_completed",
                "task_failed",
                "task_cancelled",
                "budget_exhausted",
              ].includes(event.event_type),
          )
          .map((event) => (
            <li key={event.sequence}>
              <span className="timeline-dot" />
              <div>
                <strong>{eventLabel(event)}</strong>
                <small>
                  {typeof event.payload.wall_ms === "number"
                    ? `${(event.payload.wall_ms / 1000).toFixed(1)} 秒 · `
                    : ""}
                  {event.evidence_refs.length
                    ? `${event.evidence_refs.length} 条证据`
                    : "已记录执行事件"}
                </small>
              </div>
            </li>
          ))}
      </ol>
      {running && (
        <p className="muted">
          仅显示服务端已记录的步骤；未完成的模型调用可能需要较长时间。取消在执行边界生效，已预留的访客额度不退回。
        </p>
      )}
      {task.status === "cancelled" && (
        <p className="muted">
          取消已记录，正在执行的调用可能在下一个执行边界才停止；已预留的访客额度不退回。
        </p>
      )}
      <details className="agent-trace">
        <summary>查看技术审计详情（{task.events.length} 个事件）</summary>
        <ol>
          {task.events.map((event) => (
            <li key={event.sequence}>
              <code>{event.sequence}</code>
              <b>{event.tool_name ?? event.event_type}</b>
              <span>
                {String(
                  event.payload.purpose ??
                    event.payload.status ??
                    event.payload.error_code ??
                    "",
                )}
              </span>
              <small>
                {event.evidence_refs.length
                  ? `${event.evidence_refs.length} 条证据`
                  : ""}
              </small>
            </li>
          ))}
        </ol>
      </details>
    </div>
  );
}
export function highlightText(text: string, quotes: string[]): React.ReactNode {
  const ranges: { start: number; end: number }[] = [];
  for (const quote of new Set(quotes.filter(Boolean))) {
    let from = 0;
    while (from < text.length) {
      const start = text.indexOf(quote, from);
      if (start < 0) break;
      ranges.push({ start, end: start + quote.length });
      from = start + quote.length;
    }
  }
  ranges.sort((a, b) => a.start - b.start);
  const merged: typeof ranges = [];
  for (const range of ranges) {
    const last = merged.at(-1);
    if (last && range.start <= last.end)
      last.end = Math.max(last.end, range.end);
    else merged.push({ ...range });
  }
  const nodes: React.ReactNode[] = [];
  let from = 0;
  for (const range of merged) {
    nodes.push(text.slice(from, range.start));
    nodes.push(
      <mark key={range.start}>{text.slice(range.start, range.end)}</mark>,
    );
    from = range.end;
  }
  nodes.push(text.slice(from));
  return nodes;
}
function highlightChildren(
  children: React.ReactNode,
  quotes: string[],
): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (typeof child === "string") return highlightText(child, quotes);
    if (
      React.isValidElement<{ children?: React.ReactNode }>(child) &&
      child.props.children
    )
      return React.cloneElement(
        child,
        {},
        highlightChildren(child.props.children, quotes),
      );
    return child;
  });
}
export function SourceExcerpt({
  text,
  quotes,
}: {
  text: string;
  quotes: string[];
}) {
  const [raw, setRaw] = useState(false);
  const heading = ({ children }: { children?: React.ReactNode }) => (
    <h4>{highlightChildren(children, quotes)}</h4>
  );
  return (
    <>
      <div className="excerpt-toolbar">
        <strong>引用原文片段</strong>
        <button className="secondary" onClick={() => setRaw(!raw)}>
          {raw ? "格式化阅读" : "查看原文文本"}
        </button>
      </div>
      <div className={`source-text ${raw ? "raw-source" : "formatted-source"}`}>
        {raw ? (
          text
        ) : (
          <Markdown
            remarkPlugins={[remarkGfm]}
            skipHtml
            components={{
              h1: heading,
              h2: heading,
              h3: heading,
              h4: heading,
              h5: heading,
              h6: heading,
              p: ({ children }) => <p>{highlightChildren(children, quotes)}</p>,
              li: ({ children }) => (
                <li>{highlightChildren(children, quotes)}</li>
              ),
              td: ({ children }) => (
                <td>{highlightChildren(children, quotes)}</td>
              ),
              th: ({ children }) => (
                <th>{highlightChildren(children, quotes)}</th>
              ),
            }}
          >
            {text}
          </Markdown>
        )}
      </div>
    </>
  );
}
