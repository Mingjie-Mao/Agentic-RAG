"""Prepare blind independent human review; never fill labels from model predictions."""
import hashlib
import json
from pathlib import Path
from copy import deepcopy
import secrets
import argparse

INDEPENDENT_PROTOCOL = 'external-independent-annotation-v1'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_external(suite_path, spec_path, target, *, root=Path('.')):
    """Prepare two source-only human passes; keep the key outside reviewer folders."""
    suite_path, spec_path, target, root = map(Path, (suite_path, spec_path, target, root))
    if not target.resolve().is_relative_to((root / '.runtime').resolve()):
        raise ValueError('Third-party review packets must remain under local .runtime')
    if target.exists():
        raise ValueError('External review directory exists; preserve previous work')
    suite, spec = json.loads(suite_path.read_text()), json.loads(spec_path.read_text())
    if suite.get('split') != 'external' or len({t['id'] for t in suite['tasks']}) != len(suite['tasks']):
        raise ValueError('Expected complete External suite with unique tasks')
    sources, documents, source_paths = [], {}, {}
    for doc in spec['documents']:
        aliases = []
        for version in doc['versions']:
            path = (root / version['path']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('Source path outside project')
            sid = 'S-' + secrets.token_hex(16)
            aliases.append(sid)
            sources.append({'source_id': sid, 'title': doc['title'], 'text': path.read_text(),
                            'effective_from': version.get('effective_from')})
            source_paths[sid] = {'path': version['path'], 'sha256': file_sha(path),
                                 'document_id': doc['document_id']}
        documents[doc['document_id']] = aliases
    # The source selection itself comes from upstream references. Reviewers can
    # inspect the complete catalog, not just the candidate sources for a question.
    tasks = []
    for task in suite['tasks']:
        # Package materialization already resolves upstream title-based evidence
        # into gold_documents. The upstream rows do not carry document_id.
        refs = list(dict.fromkeys(task['gold_documents']))
        if any(ref not in documents for ref in refs):
            raise ValueError('Referenced source missing from complete catalog')
        tasks.append((task, [sid for ref in refs for sid in documents[ref]]))
    packets, mappings = {}, {}
    for slot in ('reviewer-1', 'reviewer-2'):
        items, key = [], {}
        for task, refs in tasks:
            rid = 'Q-' + secrets.token_hex(16)
            item = {'id': rid, 'question': task['question'], 'candidate_source_ids': refs,
                    'annotation': {'status': 'pending', 'answerable': None, 'reference_answer': None,
                                   'required_facts': [], 'conditions': None, 'temporal_scope': None,
                                   'source_requirements': None, 'reason': None}}
            items.append(item)
            key[rid] = task['id']
        secrets.SystemRandom().shuffle(items)
        packet = {'protocol': INDEPENDENT_PROTOCOL, 'status': 'pending',
                  'reviewer': None, 'reviewer_type': None, 'independent': None,
                  'completed_at': None,
                  'declaration': {'saw_system_outputs': None, 'saw_existing_annotations': None,
                                  'saw_other_reviewer_labels': None},
                  'instructions': 'Annotate from the question and original sources. Existing answers, '
                  'model predictions, experiment methods and expected outcomes are withheld. Use '
                  'the entire source catalog for alternative evidence; candidate sources are only '
                  'upstream references, not required evidence or a correctness label. Unresolved '
                  'questions remain unclear; do not force a yes/no. Record material facts, units, '
                  'dates, subjects, speakers, specified sources and active condition branches. '
                  'Every fact cites source_id and an exact original quote.',
                  'source_selection_limit': 'Candidate pool inherited from upstream references; '
                  'not independently selected. Full corpus catalog is available.',
                  'sources': deepcopy(sources), 'items': items}
        packets[slot], mappings[slot] = packet, key
    target.mkdir(parents=True)
    bindings = {}
    for slot, packet in packets.items():
        directory = target / slot
        directory.mkdir()
        path = directory / 'packet.json'
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
        bindings[slot] = {'packet_sha256': file_sha(path), 'key': mappings[slot]}
    coordinator = target / 'coordinator'
    coordinator.mkdir()
    key = {'protocol': INDEPENDENT_PROTOCOL, 'suite_sha256': file_sha(suite_path),
           'spec_sha256': file_sha(spec_path), 'bindings': bindings, 'sources': source_paths,
           'status': 'pending', 'independent_human_labels': 0}
    (coordinator / 'key.json').write_text(json.dumps(key, indent=2) + '\n')
    (coordinator / 'original-annotations.json').write_bytes(suite_path.read_bytes())
    (coordinator / 'disagreements.json').write_text(json.dumps({
        'status': 'pending', 'reviews_received': 0, 'disagreements': None, 'adjudications': []}, indent=2) + '\n')
    return {'tasks': len(tasks), 'sources': len(sources), 'reviewer_packets': 2,
            'independent_human_labels': 0, 'status': 'pending'}


def validate_external_review(original, completed):
    """Validate human declarations and source-bound labels, never infer expertise."""
    if completed.get('protocol') != INDEPENDENT_PROTOCOL or original.get('protocol') != INDEPENDENT_PROTOCOL:
        raise ValueError('External review protocol changed')
    if (completed.get('status') != 'complete' or completed.get('reviewer_type') != 'human' or completed.get('independent') is not True
            or not str(completed.get('reviewer') or '').strip() or not completed.get('completed_at')):
        raise ValueError('Named independent human and completion time required')
    if completed.get('declaration') != dict.fromkeys(original['declaration'], False):
        raise ValueError('Human reviewer was exposed to outputs or previous judgments')
    allowed = {'reviewer', 'reviewer_type', 'independent', 'completed_at', 'declaration', 'status', 'items'}
    if set(completed) != set(original) or any(completed[k] != original[k] for k in original if k not in allowed):
        raise ValueError('Frozen review instructions or sources changed')
    expected = {r['id']: r for r in original['items']}
    ids = [r['id'] for r in completed['items']]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('Every External task must appear exactly once')
    sources = {s['source_id']: s['text'] for s in original['sources']}
    for item in completed['items']:
        before = expected[item['id']]
        if set(item) != set(before) or any(item[k] != before[k] for k in before if k != 'annotation'):
            raise ValueError('Question or source pool changed')
        annotation = item['annotation']
        if set(annotation) != set(before['annotation']) or annotation['status'] not in {'approved', 'needs_revision', 'unclear'}:
            raise ValueError('Every annotation needs an explicit disposition')
        if not str(annotation['reason'] or '').strip():
            raise ValueError('Human annotation reason required')
        if annotation['status'] == 'unclear':
            continue
        if (type(annotation['answerable']) is not bool or not isinstance(annotation['reference_answer'], str)
                or not annotation['reference_answer'].strip() or not isinstance(annotation['required_facts'], list)):
            raise ValueError('Answerability, reference answer and fact list required')
        if annotation['answerable'] and not annotation['required_facts']:
            raise ValueError('Answerable question needs material facts')
        if any(not isinstance(annotation[k], str) or not annotation[k].strip()
               for k in ('conditions', 'temporal_scope', 'source_requirements')):
            raise ValueError('Conditions, temporal scope and source requirements must be explicit; use none if inapplicable')
        for fact in annotation['required_facts']:
            if not isinstance(fact, dict) or not str(fact.get('text') or '').strip() or not fact.get('citations'):
                raise ValueError('Every fact needs source-bound evidence')
            for citation in fact['citations']:
                sid, quote = citation.get('source_id'), citation.get('quote')
                if sid not in sources or not isinstance(quote, str) or not quote or quote not in sources[sid]:
                    raise ValueError('Fact quote must be an original source span')
    return {r['id']: r['annotation'] for r in completed['items']}


def compare_external_reviews(first, second, key, *, originals):
    """Identify disagreement; agreement is not automatic adjudication or gold."""
    if first['reviewer'].strip().casefold() == second['reviewer'].strip().casefold():
        raise ValueError('Two distinct independent humans required')
    rows = []
    for slot, packet, original in zip(('reviewer-1', 'reviewer-2'), (first, second), originals, strict=True):
        raw = (json.dumps(original, ensure_ascii=False, indent=2) + '\n').encode()
        if hashlib.sha256(raw).hexdigest() != key['bindings'][slot]['packet_sha256']:
            raise ValueError('Original reviewer packet fingerprint differs')
        validate_external_review(original, packet)
        rows.append({key['bindings'][slot]['key'][r['id']]: r['annotation'] for r in packet['items']})
    if rows[0].keys() != rows[1].keys():
        raise ValueError('Reviewer coverage differs')
    dimensions = ('status', 'answerable', 'reference_answer', 'required_facts', 'conditions',
                  'temporal_scope', 'source_requirements')
    return {'status': 'awaiting_adjudication', 'reviewers': [first['reviewer'], second['reviewer']],
            'disagreements': [{'task_id': tid, 'different_fields': [f for f in dimensions if rows[0][tid][f] != rows[1][tid][f]]}
                              for tid in sorted(rows[0])],
            'adjudications': [], 'promoted_gold_labels': 0}


def prepare(source, target):
    raw = source.read_bytes()
    fixture = json.loads(raw)
    packet = {'source_sha256': hashlib.sha256(raw).hexdigest(),
              'reviewer': None, 'reviewer_type': 'human', 'status': 'pending',
              'rubric': 'Judge entailment separately from relevance. Preserve speaker, modality, negation scope, compared attribute, time and conclusion. Atomic false claims are unsupported; genuinely mixed claims may be partial.',
              'items': [{key: row[key] for key in ('id', 'category', 'question', 'evidence', 'claim')}
                        | {'entailment': None, 'relevance': None, 'reason': None}
                        for row in fixture['items']]}
    if target.exists():
        raise ValueError('Review packet exists; refusing to overwrite human work')
    target.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + '\n')
    return packet


def audited_labels(packet):
    if packet.get('reviewer_type') != 'human' or not str(packet.get('reviewer') or '').strip():
        raise ValueError('Independent human reviewer identity is required')
    source = Path(__file__).resolve().parents[1] / 'fixtures/source_contract/semantic-calibration-v4.json'
    if packet.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Human review source fingerprint differs')
    expected = {row['id']: row for row in json.loads(source.read_text())['items']}
    ids = [row.get('id') for row in packet['items']]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('Review must cover the entire frozen set exactly once')
    labels = {'supported', 'partial', 'unsupported', 'unclear'}
    for row in packet['items']:
        if any(row.get(key) != expected[row['id']][key]
               for key in ('category', 'question', 'evidence', 'claim')):
            raise ValueError('Human review changed frozen case content')
        if row.get('entailment') not in labels or row.get('relevance') not in {
            'answers', 'related', 'off_topic', 'unclear'} or not row.get('reason'):
            raise ValueError('Every item requires two labels and an explanation')
    return {row['id']: row['entailment'] for row in packet['items']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-suite', type=Path)
    parser.add_argument('--external-spec', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--original', type=Path)
    parser.add_argument('--completed', type=Path)
    args = parser.parse_args()
    if args.original or args.completed:
        if not (args.original and args.completed):
            parser.error('--original and --completed required together')
        labels = validate_external_review(json.loads(args.original.read_text()), json.loads(args.completed.read_text()))
        print(json.dumps({'reviewed': len(labels), 'approved': sum(v['status'] == 'approved' for v in labels.values()),
                          'unclear': sum(v['status'] == 'unclear' for v in labels.values()), 'promoted_gold_labels': 0}))
    elif args.external_suite:
        if not (args.external_spec and args.output):
            parser.error('--external-spec and --output required')
        print(json.dumps(prepare_external(args.external_suite, args.external_spec, args.output, root=args.root)))
    else:
        packet = prepare(Path('fixtures/source_contract/semantic-calibration-v4.json'),
                         Path('artifacts/source-human-review-p0-p2.json'))
        print(f"Prepared {len(packet['items'])} blind cases; independent human labels=0")
