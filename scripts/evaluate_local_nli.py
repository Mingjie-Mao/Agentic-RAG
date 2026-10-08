"""Evaluate a cached multilingual NLI classifier without tuning on gold labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
SOURCE = Path('.runtime/ocnli-dev.json')
MANIFEST = Path('fixtures/source_contract/ocnli-hard-gov-news-v1.json')
OUT = Path('artifacts/ocnli-local-nli-v1.json')
AUTHORED = Path('fixtures/source_contract/semantic-calibration-v4.json')
AUTHORED_OUT = Path('artifacts/p0-p2-authored-local-nli-v1.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def classify(model, tokenizer, torch, pairs):
    predictions = []
    started = time.monotonic()
    for index in range(0, len(pairs), 8):
        batch = pairs[index:index + 8]
        encoded = tokenizer([row[0] for row in batch], [row[1] for row in batch],
                            padding=True, truncation=True, max_length=384, return_tensors='pt')
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
        for vector in probabilities:
            by_name = {model.config.id2label[i].casefold(): round(value, 4)
                       for i, value in enumerate(vector)}
            predictions.append({'predicted': max(by_name, key=by_name.get), 'probabilities': by_name})
    return predictions, round((time.monotonic() - started) * 1000, 1)


def report(gold, predictions):
    rows = [{**row, **prediction} for row, prediction in zip(gold, predictions, strict=True)]
    supported = [row for row in rows if row['gold'] == 'entailment']
    other = [row for row in rows if row['gold'] != 'entailment']
    return rows, {
        'n': len(rows), 'by_gold': dict(Counter(row['gold'] for row in rows)),
        'three_way_correct': sum(row['gold'] == row['predicted'] for row in rows),
        'supported_recall': round(sum(row['predicted'] == 'entailment' for row in supported) / len(supported), 3),
        'not_supported_recall': round(sum(row['predicted'] != 'entailment' for row in other) / len(other), 3),
        'false_accepts': sum(row['predicted'] == 'entailment' for row in other),
        'false_rejects': sum(row['predicted'] != 'entailment' for row in supported),
        'confusion': {label: dict(Counter(row['predicted'] for row in rows if row['gold'] == label))
                      for label in ('entailment', 'neutral', 'contradiction')},
    }


def main():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if OUT.exists() or AUTHORED_OUT.exists():
        raise SystemExit('Refusing to overwrite NLI outputs')
    torch.set_num_threads(4)
    snapshot = snapshot_download(MODEL, local_files_only=True, allow_patterns=[
        'config.json', 'model.safetensors', 'spm.model', 'tokenizer.json',
        'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json'])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True).eval()
    names = {name.casefold() for name in model.config.id2label.values()}
    if names != {'entailment', 'neutral', 'contradiction'}:
        raise SystemExit(f'Unexpected NLI label order: {model.config.id2label}')
    metadata = {'model': MODEL, 'model_revision': Path(snapshot).name,
                'weights_sha256': sha(Path(snapshot) / 'model.safetensors'),
                'runtime': 'local_cpu', 'tuning': 'none', 'project_case_human_reviewed': 0}

    manifest = json.loads(MANIFEST.read_text())
    if sha(SOURCE) != manifest['source_sha256']:
        raise SystemExit('Human-labeled source changed')
    source = {row['id']: row for row in (json.loads(line) for line in SOURCE.open())}
    chosen = [source[item['id']] for item in manifest['items']]
    pairs = [(row['sentence1'], row['sentence2']) for row in chosen]
    rows, elapsed = classify(model, tokenizer, torch, pairs)
    labeled, summary = report([{'id': row['id'], 'genre': row['genre'], 'gold': row['label']}
                               for row in chosen], rows)
    OUT.write_text(json.dumps({'metadata': {**metadata, 'manifest_sha256': sha(MANIFEST),
                                            'source': 'OCNLI dev human-majority labels; external NLI only'},
                               'summary': {**summary, 'elapsed_ms': elapsed}, 'records': labeled},
                              ensure_ascii=False, indent=2) + '\n')
    print('OCNLI', json.dumps({**summary, 'elapsed_ms': elapsed}, ensure_ascii=False), flush=True)

    authored = json.loads(AUTHORED.read_text())['items']
    predicted, authored_elapsed = classify(model, tokenizer, torch,
                                           [(row['evidence'], row['claim']) for row in authored])
    # The authored set has no neutral label. Map its unsupported/partial distinction
    # only to binary supported-versus-not-supported for this NLI model.
    authored_gold = [{'id': row['id'], 'category': row['category'],
                      'gold': 'entailment' if row['expected'] == 'supported' else 'contradiction'}
                     for row in authored]
    labeled, authored_summary = report(authored_gold, predicted)
    AUTHORED_OUT.write_text(json.dumps({
        'metadata': {**metadata, 'fixture_sha256': sha(AUTHORED),
                     'source': 'Previously used assistant-authored development labels; not independent human'},
        'summary': {**authored_summary, 'elapsed_ms': authored_elapsed}, 'records': labeled,
    }, ensure_ascii=False, indent=2) + '\n')
    print('AUTHORED', json.dumps({**authored_summary, 'elapsed_ms': authored_elapsed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
