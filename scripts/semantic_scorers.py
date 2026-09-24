"""P2: score every labelled claim with four semantic scorers, without reading any label.

The scorers see exactly what the annotators saw — the question, the claim and the
cited evidence from `claims-to-label.csv` — and nothing else: no gold, no system
verdict. Predictions are cached so that the comparison against the frozen gold
(`evaluate_semantic_scorers.py`) can never feed back into how they were produced.

Cost ladder, cheapest first: token overlap, bge-m3 cosine, bge-reranker cross-encoder,
qwen2.5 LLM judge. See `artifacts/semantic-calibration/PREREGISTRATION-P2.md`.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import time

DIR = Path(__file__).resolve().parent.parent / "artifacts" / "semantic-calibration"

STOPWORDS = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    this that these those it its he she they them his her their there here which who whom whose what
    when where why how not no nor do does did has have had will would can could should may might must
    also than then so such about into over after before between both each other some any all more most
    only own same very just while however article articles report reports reported according""".split()
)
WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
CJK = re.compile(r"[一-鿿]+")
WINDOW, STEP = 1200, 800

LLM_RULES = """你是语义蕴含标注员。只依据给出的 evidence 判断，不用任何外部知识。

entailment（claim 是否被 evidence 支持）：
- supported：evidence 足以推出 claim 的全部实质断言。换了措辞、概括得更粗仍算 supported。
- partial：claim 含多个实质断言，evidence 只支持其中一部分。
- unsupported：evidence 推不出、与 evidence 冲突，或需要关键外部信息。
补充规则：
- claim 里复述的问题限定语（事件、数字、对象）也是断言，照样要被支持。
- 「据 X 报道」这类来源归属、文档头里的日期作者，不算实质断言。
- evidence 写「据称 / 被指控」而 claim 写成确定事实 → unsupported。
- 「文章没有提到 X」这类否定式断言：evidence 不冲突 → partial；evidence 明确说了 X → unsupported。
- 关系断言（一致 / 不一致 / 有变化 / 相同 / 不同）：两侧事实都在且足以推出 → supported；
  两侧都在但推不出，或任一侧缺失 → unsupported。

relevance（claim 是否回答了 question 所问）：
- answers：直接回答问题要求（合取问句中回答其中一支也算；wh 问句中以答案实体为主语的断言也算）。
- related：有关但没有回答所问（例如比较问句里只陈述一侧、不下关系判断）。
- off_topic：基本无关。

只输出指定 JSON。"""

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "entailment": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
        "relevance": {"type": "string", "enum": ["answers", "related", "off_topic"]},
    },
    "required": ["entailment", "relevance"],
}


def content_terms(text):
    text = (text or "").lower()
    words = {w for w in WORD.findall(text) if w not in STOPWORDS and (len(w) > 2 or w.isdigit())}
    grams = set()
    for run in CJK.findall(text):
        grams |= {run[i : i + 2] for i in range(max(0, len(run) - 1))}
    return words | grams


def blocks(evidence):
    parts = [part.strip() for part in re.split(r"\n-{3,}\n", evidence or "") if part.strip()]
    return parts or [""]


def windows(evidence):
    """Evidence blocks cut into overlapping windows, so a 512-token model sees all of it."""
    found = []
    for block in blocks(evidence):
        if len(block) <= WINDOW:
            found.append(block)
            continue
        found.extend(block[start : start + WINDOW] for start in range(0, len(block) - STEP, STEP))
    return found


def rule_scorer():
    def score(premise, hypothesis):
        wanted = content_terms(hypothesis)
        return len(wanted & content_terms(premise)) / len(wanted) if wanted else 0.0

    return score


def embedding_scorer():
    from app.clients import Models

    models = Models()

    def cosine(left, right):
        norm = sum(x * x for x in left) ** 0.5 * sum(y * y for y in right) ** 0.5
        return sum(x * y for x, y in zip(left, right, strict=True)) / norm if norm else 0.0

    def score(premise, hypothesis):
        passages = windows(premise)
        vectors = models.embed([hypothesis, *passages])
        return max(cosine(vectors[0], vector) for vector in vectors[1:])

    return score


def cross_encoder_scorer():
    from app.rerank import rerank

    def score(premise, hypothesis):
        ranked = rerank(hypothesis, [{"text": text} for text in windows(premise)])
        return max(item["rerank_score"] for item in ranked)

    return score


def llm_judge():
    from app.clients import Models
    from app.config import settings

    models = Models()

    def judge(question, claim, evidence):
        result = models._post(
            "/api/chat",
            {
                "model": settings().chat_model,
                "stream": False,
                "keep_alive": "30m",
                "format": LLM_SCHEMA,
                "messages": [
                    {"role": "system", "content": LLM_RULES},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "claim": claim, "evidence": evidence[:12000]},
                            ensure_ascii=False,
                        ),
                    },
                ],
                "options": {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 40},
            },
        )
        label = json.loads(result["message"]["content"])
        return label, {
            "prompt_tokens": result.get("prompt_eval_count", 0),
            "completion_tokens": result.get("eval_count", 0),
        }

    return judge


def load_claims():
    with (DIR / "claims-to-label.csv").open(encoding="utf-8-sig") as handle:
        return [
            {key: row[key] for key in ("item_id", "claim_index", "question", "claim", "cited_evidence")}
            for row in csv.DictReader(handle)
        ]


def timed(function, *args):
    started = time.perf_counter()
    value = function(*args)
    return value, round((time.perf_counter() - started) * 1000, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorers", default="rule,embedding,cross_encoder,llm_judge")
    parser.add_argument("--out", default=str(DIR / "scorer-predictions.json"))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟）")
    args = parser.parse_args()

    claims = load_claims()[: args.limit or None]
    out = Path(args.out)
    cached = json.loads(out.read_text()) if out.exists() else {"predictions": {}}
    predictions = cached["predictions"]
    factories = {"rule": rule_scorer, "embedding": embedding_scorer, "cross_encoder": cross_encoder_scorer}

    for name in args.scorers.split(","):
        # Re-running a scorer replaces its predictions whole, never half-merges two runs.
        runs = {}
        if name == "llm_judge":
            judge = llm_judge()
            for number, row in enumerate(claims, 1):
                (label, usage), elapsed = timed(judge, row["question"], row["claim"], row["cited_evidence"])
                runs[f"{row['item_id']}#{row['claim_index']}"] = {**label, **usage, "ms": elapsed}
                if number % 20 == 0:
                    print(f"  {name} {number}/{len(claims)}", flush=True)
        else:
            score = factories[name]()
            for row in claims:
                entail, entail_ms = timed(score, row["cited_evidence"], row["claim"])
                relevance, relevance_ms = timed(score, row["question"], row["claim"])
                runs[f"{row['item_id']}#{row['claim_index']}"] = {
                    "entailment": round(entail, 4),
                    "relevance": round(relevance, 4),
                    "ms": entail_ms,
                    "relevance_ms": relevance_ms,
                }
        predictions[name] = runs
        print(f"{name}: {len(runs)} 条", flush=True)

    cached.update(
        {
            "source": "artifacts/semantic-calibration/claims-to-label.csv",
            "labels_read": "none — 预测阶段不读取任何标签",
            "window_chars": WINDOW,
            "window_step": STEP,
            "llm_rules": LLM_RULES,
            "predictions": predictions,
        }
    )
    out.write_text(json.dumps(cached, ensure_ascii=False, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    main()
