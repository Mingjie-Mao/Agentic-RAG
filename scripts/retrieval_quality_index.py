"""Isolated keyword context enrichment; original embeddings and PostgreSQL stay intact."""

from collections import Counter
import json
import math
import time

from sqlalchemy import select

from app.boilerplate import is_web_boilerplate
from app.ingestion import chunk_header, indexed_text
from app.models import Chunk, DocumentVersion, User
from app.security import readable_documents, require_chunk
from scripts.research_review import digest

DEV_USERS = ('lt-eng', 'mh-eval')


def active_scope(db):
    """Snapshot every active readable chunk, including all distractors, for both users."""
    db.expire_all()
    users, documents, records, bindings = {}, {}, [], {}
    for uid in DEV_USERS:
        user = db.get(User, uid, populate_existing=True)
        if user is None or not user.active:
            raise ValueError('Dev user missing or inactive')
        users[uid] = user
        readable = readable_documents(db, user)
        bindings[uid] = sorted(d.id for d in readable if d.active_version_id)
        documents.update((d.id, d) for d in readable if d.active_version_id)
    ledger = []
    for did, doc in sorted(documents.items()):
        version = db.get(DocumentVersion, doc.active_version_id, populate_existing=True)
        if version is None or version.document_id != did or version.status != 'ready':
            raise ValueError('active version unavailable')
        chunks = db.scalars(select(Chunk).where(Chunk.version_id == version.id)
                            .order_by(Chunk.ordinal)).all()
        for chunk in chunks:
            reader = next(users[uid] for uid in DEV_USERS if did in bindings[uid])
            require_chunk(db, reader, chunk.id, active_only=True)
            records.append((chunk, version, doc))
        ledger.append({'document_id': did, 'tenant_id': doc.tenant_id,
            'active_version_id': version.id, 'source_sha256': version.content_hash,
            'acl_sha256': digest([doc.owner_id, sorted(doc.read_groups), doc.tenant_public,
                                  doc.deleted, doc.revision]),
            'document_sha256': digest([doc.title, doc.metadata_json]),
            'pipeline_sha256': digest(version.pipeline),
            'chunks': [{'id': c.id, 'ordinal': c.ordinal,
                        'text_sha256': digest(c.text), 'locator_sha256': digest(c.locator)} for c in chunks]})
    user_ledger = [{'id': u.id, 'tenant_id': u.tenant_id, 'identity_sha256': digest(
        [u.username, u.role, sorted(u.groups), u.active]), 'readable_document_ids': bindings[uid]}
        for uid, u in sorted(users.items())]
    return users, records, {'documents': ledger, 'users': user_ledger}


def source_records(search, records):
    """Validate exact source identities and vector availability before any target write."""
    by_id = {c.id: (c, v, d) for c, v, d in records}
    if len(by_id) != len(records):
        raise ValueError('duplicate source chunk IDs')
    mapping = search.request('GET', f'/{search.index}/_mapping')
    dimension = mapping[search.index]['mappings']['properties']['embedding']['dimension']
    found = {}
    ids = sorted(by_id)
    for offset in range(0, len(ids), 256):
        batch = ids[offset:offset + 256]
        rows = search.request('POST', f'/{search.index}/_mget', json={'ids': batch})['docs']
        if len(rows) != len(batch) or {r.get('_id') for r in rows} != set(batch):
            raise ValueError('source response IDs mismatch')
        for row in rows:
            if not row.get('found') or row['_id'] in found:
                raise ValueError('source chunk missing or duplicated')
            src = row['_source']
            chunk, version, doc = by_id[row['_id']]
            if (src.get('tenant_id'), src.get('document_id'), src.get('version_id')) != (
                    doc.tenant_id, doc.id, version.id):
                raise ValueError('source chunk scope mismatch')
            vector = src.get('embedding')
            if (not isinstance(vector, list) or len(vector) != dimension
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)):
                raise ValueError('source vector missing or invalid')
            header = chunk_header(doc.title, doc.metadata_json, {'chunk_context': 'document_header'})
            if src.get('text') not in (chunk.text, indexed_text(header, chunk.text)):
                raise ValueError('source indexed text mismatch')
            found[row['_id']] = src
    return found, mapping


def normalized_settings(settings):
    """Canonical scalar/flattened settings, excluding only index identity metadata."""
    flat = {}
    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f'{prefix}.{key}' if prefix else key)
        else:
            name = prefix.removeprefix('index.')
            identity = {'uuid', 'provided_name', 'creation_date', 'creation_date_string', 'history_uuid'}
            if name in identity or name.startswith(('version.', 'resize.source.')):
                return
            def scalar(item):
                return str(item).lower() if isinstance(item, bool) else str(item) if isinstance(
                    item, (int, float)) else item
            flat[name] = [scalar(item) for item in value] if isinstance(value, list) else scalar(value)
    visit(settings)
    return flat


def settings_record(search):
    item = search.request('GET', f'/{search.index}/_settings?include_defaults=true')[search.index]
    explicit = item['settings']['index']
    defaults = item.get('defaults', {})
    defaults = defaults.get('index', defaults)
    effective = normalized_settings(defaults) | normalized_settings(explicit)
    return explicit, effective


def index_population(search):
    settings, effective = settings_record(search)
    count = search.request('GET', f'/{search.index}/_count')['count']
    stats = search.request('GET', f'/{search.index}/_stats/indexing?filter_path=indices.*.primaries.indexing.index_total,indices.*.primaries.indexing.delete_total')
    return {'count': count, 'settings_sha256': digest(settings),
            'effective_settings_sha256': digest(effective), 'sequence_sha256': digest(stats)}


def index_ledger(search, records):
    found, mapping = source_records(search, records)
    return {'index': search.index, 'population': index_population(search), 'mapping_sha256': digest(mapping[search.index]['mappings']),
            'chunks': [{'id': cid, 'source_sha256': digest(src),
                        'text_sha256': digest(src['text']), 'vector_sha256': digest(src['embedding'])}
                       for cid, src in sorted(found.items())]}


def build_keyword_index(source, target, records, *, on_progress=None):
    """Create a fresh index after complete validation; failed indexes are retained."""
    if source.index == target.index:
        raise ValueError('keyword experiment cannot target production index')
    existing = target.request('GET', '/_cat/indices?format=json')
    if target.index in {row['index'] for row in existing}:
        raise ValueError('keyword experiment target already exists')
    found, mapping = source_records(source, records)
    source_settings, source_effective_settings = settings_record(source)
    baseline_population = index_population(source)
    kept, removed, changed = [], [], 0
    for chunk, version, doc in records:
        src = found[chunk.id]
        if is_web_boilerplate(chunk.text):
            removed.append(chunk.id)
            continue
        header = chunk_header(doc.title, doc.metadata_json, {'chunk_context': 'document_header'})
        enriched = indexed_text(header, chunk.text)
        changed += enriched != src['text']
        kept.append((chunk, version, doc, src['embedding'], enriched))
    # An exclusive create is intentional: ensure_index permits existing indexes and
    # would leave a check/create race. OpenSearch rejects PUT when the name exists.
    target.request('PUT', f'/{target.index}', json={
        'settings': {'index': normalized_settings(source_settings)},
        'mappings': mapping[source.index]['mappings']})
    # Copy the full index inside OpenSearch. Pruning inactive or unrelated tenant
    # docs would change BM25 document-frequency statistics and confound enrichment.
    # No unreadable source text leaves the search engine.
    started = target.request('POST', '/_reindex?wait_for_completion=false&refresh=true',
        json={'source': {'index': source.index}, 'dest': {'index': target.index, 'op_type': 'create'}})
    task_id = started.get('task')
    if not isinstance(task_id, str) or not task_id:
        raise ValueError('official reindex task identity unavailable')
    progress = {'status': 'reindex_running', 'source_index': source.index,
                'target_index': target.index, 'official_task_id': task_id}
    if on_progress:
        on_progress(progress)  # Durable local callback runs before any long wait.
    while True:
        task = target.request('GET', f'/_tasks/{task_id}')
        if not task.get('completed'):
            time.sleep(1)
            continue
        if task.get('error'):
            if on_progress:
                on_progress(progress | {'status': 'reindex_failed', 'failure_count': 1,
                    'error': {'type': 'OpenSearchTaskFailure', 'stage': 'reindex'}})
            raise ValueError('official source reindex task failed')
        copied = task.get('response', {})
        break
    if (copied.get('failures') or copied.get('timed_out') or copied.get('version_conflicts')
            or copied.get('total') != baseline_population['count']
            or copied.get('created') != baseline_population['count']):
        if on_progress:
            on_progress(progress | {'status': 'reindex_failed',
                'failure_count': len(copied.get('failures') or []),
                'error': {'type': 'IncompleteReindex', 'stage': 'reindex'}})
        raise ValueError('full source index copy incomplete')
    if on_progress:
        on_progress(progress | {'status': 'reindex_complete', 'failure_count': 0,
            'reindex_total': copied['total'], 'reindex_created': copied['created']})
    grouped = {}
    for row in kept:
        if row[4] != found[row[0].id]['text']:
            grouped.setdefault(row[2].id, []).append(row)
    for rows in grouped.values():
        _, version, doc, _, _ = rows[0]
        target.index_chunks(doc.tenant_id, doc.id, version.id, [r[0] for r in rows],
                            [r[3] for r in rows], title=doc.title, texts=[r[4] for r in rows])
    if removed:
        body = ''.join(json.dumps({'delete': {'_index': target.index, '_id': cid}}) + '\n'
                       for cid in removed)
        result = target.request('POST', '/_bulk?refresh=wait_for', content=body,
                                headers={'Content-Type': 'application/x-ndjson'})
        if result.get('errors'):
            raise ValueError('keyword boilerplate deletion incomplete')
    if index_population(source) != baseline_population:
        raise ValueError('source population changed during clone')
    return {'name': 'keyword context enrichment', 'status': 'written_pending_verification',
            'source_index': source.index, 'target_index': target.index,
            'official_task_id': task_id, 'failure_count': 0,
            'full_population_policy': 'server_side_full_reindex_then_authorized_scope_transform',
            'baseline_population': baseline_population,
            'reindex_total': copied['total'], 'reindex_created': copied['created'],
            'untouched_population_count': baseline_population['count'] - len(records),
            'expected_target_population_count': baseline_population['count'] - len(removed),
            'source_document_count': len({d.id for _, _, d in records}),
            'source_chunk_count': len(records), 'indexed_chunk_count': len(kept),
            'removed_boilerplate_chunks': len(removed), 'removed_chunk_ids': sorted(removed),
            'changed_texts': changed, 'vector_policy': 'original_vectors_unchanged',
            'tenant_document_counts': dict(Counter(d.tenant_id for d in
                {d.id: d for _, _, d in records}.values())),
            'source_mapping_sha256': digest(mapping[source.index]['mappings']),
            'source_effective_settings_sha256': digest(source_effective_settings),
            'source_ledger': [{'id': cid, 'source_sha256': digest(src),
                               'vector_sha256': digest(src['embedding'])}
                              for cid, src in sorted(found.items())],
            'target_ledger': [{'id': c.id, 'document_id': d.id, 'version_id': v.id,
                'text_sha256': digest(text), 'vector_sha256': digest(vector)}
                for c, v, d, vector, text in sorted(kept, key=lambda r: r[0].id)]}


def verify_keyword_index(source, target, records, record):
    """Read back all target chunks and exact vectors; no partially built success record."""
    omitted = set(record['removed_chunk_ids'])
    kept = [r for r in records if r[0].id not in omitted]
    ledger = index_ledger(target, kept)
    if (ledger['mapping_sha256'] != record['source_mapping_sha256']
            or ledger['population']['effective_settings_sha256'] != record['source_effective_settings_sha256']):
        raise ValueError('keyword target configuration differs from source clone expectation')
    expected = {r['id']: r for r in record['target_ledger']}
    for row in ledger['chunks']:
        if any(row[k] != expected[row['id']][k] for k in ('text_sha256', 'vector_sha256')):
            raise ValueError('keyword target readback mismatch')
    count = target.request('GET', f'/{target.index}/_count')['count']
    if omitted:
        absent = target.request('POST', f'/{target.index}/_mget', json={'ids': sorted(omitted)})['docs']
        if len(absent) != len(omitted) or {r.get('_id') for r in absent} != omitted or any(
                r.get('found') for r in absent):
            raise ValueError('removed boilerplate remains in target')
    if count != record['expected_target_population_count']:
        raise ValueError('keyword target contains unexpected chunks')
    return dict(record, status='complete', index_identity=ledger,
                index_identity_sha256=digest(ledger))
