"""Fine-tune the cross-encoder into a supporting-sentence selector on HotpotQA.

The general reranker picks the sentences that look most like the question; a yes/no
comparison needs the sentences a judgement depends on. HotpotQA labels exactly those
("supporting facts") for every question, on Wikipedia — no overlap with the 2023 news
corpus the external evaluation uses, so nothing about that evaluation leaks in.

Training is listwise: for each supporting sentence, one group of that sentence plus
seven negatives (other sentences of the gold paragraphs first — the hard ones — then
distractor sentences), cross-entropy over the group. Comparison questions are
oversampled to half the training set because that is the question type the system
fails on. Inputs are "title\\nsentence", the same shape the selector scores at runtime.

Data: HotpotQA distractor (CC BY-SA 4.0) converted to JSONL under .runtime/hotpotqa/.
The fine-tuned model is written to .runtime/models/ and never committed.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".runtime/huggingface"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

DATA = ROOT / ".runtime/hotpotqa"
BASE = "BAAI/bge-reranker-v2-m3"
OUT = ROOT / os.environ.get("SELECTOR_OUT", ".runtime/models/sentence-selector-v1")
REPORT = ROOT / os.environ.get("SELECTOR_REPORT", "artifacts/sentence-selector-training.json")


def load(split):
    # split("\n"), not splitlines(): Wikipedia text contains U+2028, which splitlines
    # treats as a line break and would cut a record in two.
    text = (DATA / f"{split}.jsonl").read_text()
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def sentences_of(record):
    rows = []
    for title, sentences in record["context"]:
        for index, sentence in enumerate(sentences):
            if sentence.strip():
                rows.append((title, index, sentence.strip()))
    return rows


def groups_for(record, rng, size):
    gold_titles = {title for title, _ in record["supporting"]}
    supporting = {(title, index) for title, index in record["supporting"]}
    rows = sentences_of(record)
    positives = [r for r in rows if (r[0], r[1]) in supporting]
    hard = [r for r in rows if r[0] in gold_titles and (r[0], r[1]) not in supporting]
    easy = [r for r in rows if r[0] not in gold_titles]
    groups = []
    for positive in positives:
        negatives = rng.sample(hard, min(len(hard), (size - 1) // 2))
        negatives += rng.sample(easy, min(len(easy), size - 1 - len(negatives)))
        if len(negatives) == size - 1:
            groups.append([positive, *negatives])
    return groups


def pair_text(row):
    return f"{row[0]}\n{row[2]}"


def evaluate(model, tokenizer, device, records, batch=32):
    """Supporting-fact recall within the top-k sentences, k = number of supporting facts."""
    model.eval()
    hits = total = all_found = 0
    at4 = 0
    with torch.no_grad():
        for record in records:
            rows = sentences_of(record)
            scores = []
            for start in range(0, len(rows), batch):
                chunk = rows[start : start + batch]
                encoded = tokenizer(
                    [record["question"]] * len(chunk), [pair_text(r) for r in chunk],
                    padding=True, truncation=True, max_length=192, return_tensors="pt",
                ).to(device)
                scores.extend(model(**encoded).logits.view(-1).float().tolist())
            ranked = [rows[i] for i in sorted(range(len(rows)), key=lambda i: -scores[i])]
            supporting = {(t, i) for t, i in record["supporting"]}
            k = len(supporting)
            top_k = {(r[0], r[1]) for r in ranked[:k]}
            top_4 = {(r[0], r[1]) for r in ranked[:4]}
            hits += len(top_k & supporting)
            total += k
            all_found += supporting <= top_4
            at4 += len(top_4 & supporting)
    model.train()
    return {
        "questions": len(records),
        "supporting_recall_at_k": round(hits / total, 4),
        "supporting_recall_at_4": round(at4 / total, 4),
        "all_supporting_in_top_4": round(all_found / len(records), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=3000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups-per-step", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eval-questions", type=int, default=400)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train = load("train")
    comparison = [r for r in train if r["type"] == "comparison"]
    bridge = [r for r in train if r["type"] != "comparison"]
    half = args.questions // 2
    chosen = rng.sample(comparison, min(half, len(comparison))) + rng.sample(bridge, args.questions - half)
    batches = [(record, g) for record in chosen for g in groups_for(record, rng, args.group_size)]
    rng.shuffle(batches)
    validation = rng.sample(load("validation"), args.eval_questions)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, local_files_only=True).to(device)
    # The 250k-token embedding matrix is almost half the parameters and carries little
    # of what "which sentence does a judgement need" is about. Freezing it halves the
    # gradient and optimizer memory, which is what lets this train on a 32 GB laptop.
    for parameter in model.get_input_embeddings().parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable()

    started = time.monotonic()
    before = evaluate(model, tokenizer, device, validation)
    print("before", before, f"{time.monotonic() - started:.0f}s", flush=True)

    steps = math.ceil(len(batches) / args.groups_per_step)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup))
    )
    model.train()
    losses = []
    for step in range(steps):
        part = batches[step * args.groups_per_step : (step + 1) * args.groups_per_step]
        questions = [record["question"] for record, group in part for _ in group]
        texts = [pair_text(row) for _record, group in part for row in group]
        encoded = tokenizer(
            questions, texts, padding=True, truncation=True, max_length=160, return_tensors="pt"
        ).to(device)
        logits = model(**encoded).logits.view(len(part), args.group_size)
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(part), dtype=torch.long, device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            recent = sum(losses[-50:]) / 50
            print(f"step {step + 1}/{steps} loss {recent:.4f} {time.monotonic() - started:.0f}s", flush=True)

    after = evaluate(model, tokenizer, device, validation)
    print("after", after, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    REPORT.write_text(
        json.dumps(
            {
                "base_model": BASE,
                "data": "HotpotQA distractor (CC BY-SA 4.0), Wikipedia; no overlap with the news corpus",
                "training_questions": len(chosen),
                "comparison_share": 0.5,
                "groups": len(batches),
                "group_size": args.group_size,
                "steps": steps,
                "frozen": "input embeddings",
                "trainable_parameters": sum(p.numel() for p in trainable),
                "lr": args.lr,
                "seed": args.seed,
                "validation_questions": len(validation),
                "validation_before": before,
                "validation_after": after,
                "first_50_loss": round(sum(losses[:50]) / min(50, len(losses)), 4),
                "last_50_loss": round(sum(losses[-50:]) / min(50, len(losses)), 4),
                "seconds": round(time.monotonic() - started),
                "model_path": str(OUT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    print(REPORT)


if __name__ == "__main__":
    main()
