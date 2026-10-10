import copy
from types import SimpleNamespace

import pytest

from scripts.retrieval_quality_index import build_keyword_index, verify_keyword_index


class Search:
    def __init__(self, index, docs=None, exists=False):
        self.index, self.docs, self.exists = index, docs or [], exists
        self.calls, self.written = [], []
        self.population = 10
        self.clone_docs = copy.deepcopy(self.docs)
        self.unscoped_untouched = False
        self.defaults = {}
        self.settings = {'knn': True, 'number_of_shards': '1', 'uuid': index + '-uuid'}
        self.mapping = {'properties': {'embedding': {'dimension': 2},
                                      'text': {'type': 'text', 'analyzer': 'cjk'}}}
    def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == '/_cat/indices?format=json':
            return [{'index': self.index}] if self.exists else []
        if path.endswith('/_mget'):
            docs = {r['_id']: r for r in self.docs}
            return {'docs': [copy.deepcopy(docs[cid]) if cid in docs else
                             {'_id': cid, 'found': False} for cid in kwargs['json']['ids']]}
        if path.split('?')[0].endswith('/_settings'):
            return {self.index: {'settings': {'index': copy.deepcopy(self.settings)},
                                 'defaults': {'index': copy.deepcopy(self.defaults)}}}
        if '/_stats' in path:
            return {}
        if path.endswith('/_count'):
            return {'count': self.population}
        if path.startswith('/_reindex?'):
            assert kwargs['json'] == {'source': {'index': 'production'},
                     'dest': {'index': 'isolated', 'op_type': 'create'}}
            self.unscoped_untouched = True
            self.docs = copy.deepcopy(self.clone_docs)
            assert path == '/_reindex?wait_for_completion=false&refresh=true'
            return {'task': 'local-node:42'}
        if path.startswith('/_tasks/'):
            assert path == '/_tasks/local-node:42'
            return {'completed': True, 'response': {'total': 10, 'created': 10, 'failures': []}}
        if path.startswith('/_bulk?'):
            assert kwargs['content'] == '{"delete": {"_index": "isolated", "_id": "c1"}}\n'
            self.population -= 1
            self.docs = [r for r in self.docs if r['_id'] != 'c1']
            return {'errors': False}
        if path.endswith('/_mapping'):
            return {self.index: {'mappings': copy.deepcopy(self.mapping)}}
        if method == 'PUT':
            self.settings = copy.deepcopy(kwargs['json']['settings']['index']) | {
                'uuid': self.index + '-uuid'}
            self.mapping = copy.deepcopy(kwargs['json']['mappings'])
            return {'acknowledged': True}
        raise AssertionError((method, path))
    def index_chunks(self, tenant, document, version, chunks, vectors, title='', texts=None):
        self.written.extend(zip([c.id for c in chunks], vectors, texts, strict=True))
        for chunk, vector, value in zip(chunks, vectors, texts, strict=True):
            row = next(r for r in self.docs if r['_id'] == chunk.id)
            row['_source'].update(embedding=vector, text=value, title=title)


def fixture():
    doc = SimpleNamespace(id='doc', tenant_id='tenant', title='Title', metadata_json={'source': 'Paper'})
    version = SimpleNamespace(id='v', content_hash='sha')
    texts = ['Article facts and details. Advertisement subscribe to our newsletter.',
             'Subscribe to our newsletter', 'Already enriched article facts.']
    records = [(SimpleNamespace(id=f'c{i}', text=text), version, doc) for i, text in enumerate(texts)]
    source_docs = [dict(_id=c.id, found=True, _source=dict(tenant_id='tenant', document_id='doc',
        version_id='v', embedding=[0.1, 0.2], text=('Title · Paper\n' + c.text if i == 2 else c.text)))
        for i, (c, v, d) in enumerate(records)]
    return records, source_docs


def test_isolated_enrichment_preserves_ids_vectors_and_mixed_text():
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    target.clone_docs = copy.deepcopy(docs)
    result = build_keyword_index(source, target, records)
    assert all(method == 'GET' or path.endswith('/_mget') for method, path, _ in source.calls)
    assert [row[0] for row in target.written] == ['c0']
    assert all(row[1] == [0.1, 0.2] for row in target.written)
    assert records[0][0].text in target.written[0][2]
    assert docs[2]['_source']['text'].count('Title · Paper') == 1
    assert result['removed_boilerplate_chunks'] == 1 and result['changed_texts'] == 1
    assert 'Article facts' not in str(result)
    assert result['vector_policy'] == 'original_vectors_unchanged'
    assert target.unscoped_untouched
    assert result['untouched_population_count'] == 7
    assert result['expected_target_population_count'] == 9
    create = next(kwargs for method, path, kwargs in target.calls if method == 'PUT')
    assert 'uuid' not in create['json']['settings']['index']


@pytest.mark.parametrize('kind', ['same', 'exists', 'missing', 'ids', 'scope', 'vector'])
def test_unsafe_index_rejected_before_writes(kind):
    records, docs = fixture()
    if kind == 'missing':
        docs[0]['found'] = False
    elif kind == 'ids':
        docs[0]['_id'] = 'wrong'
    elif kind == 'scope':
        docs[0]['_source']['tenant_id'] = 'wrong'
    elif kind == 'vector':
        docs[0]['_source'].pop('embedding')
    source = Search('production', docs)
    target = Search('production' if kind == 'same' else 'isolated', exists=kind == 'exists')
    with pytest.raises(ValueError):
        build_keyword_index(source, target, records)
    assert not target.written
    assert not any(method == 'PUT' for method, _, _ in target.calls)


@pytest.mark.parametrize('mutation', ['none', 'vector', 'text', 'extra'])
def test_readback_required_before_completed_record(mutation):
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    target.clone_docs = copy.deepcopy(docs)
    record = build_keyword_index(source, target, records)
    assert record['status'] == 'written_pending_verification'
    if mutation == 'vector':
        target.docs[0]['_source']['embedding'] = [0.2, 0.3]
    elif mutation == 'text':
        target.docs[0]['_source']['text'] = 'unexpected unrelated passage'
    elif mutation == 'extra':
        target.population += 1
    if mutation == 'none':
        verified = verify_keyword_index(source, target, records, record)
        assert verified['status'] == 'complete'
        assert verified['index_identity']['population']['count'] == 9
    else:
        with pytest.raises(ValueError):
            verify_keyword_index(source, target, records, record)


def test_reindex_partial_copy_retained_without_scope_transform():
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    request = target.request
    def partial(method, path, **kwargs):
        if path.startswith('/_tasks/'):
            return {'completed': True, 'response': {'total': 10, 'created': 9,
                    'failures': [{'reason': 'private text'}]}}
        return request(method, path, **kwargs)
    target.request = partial
    with pytest.raises(ValueError, match='copy incomplete'):
        build_keyword_index(source, target, records)
    assert any(method == 'PUT' for method, _, _ in target.calls)
    assert not target.written
    assert not any(method == 'DELETE' for method, _, _ in target.calls)


def test_already_enriched_scope_is_honest_noop():
    records, docs = fixture()
    source, target = Search('production', [docs[2]]), Search('isolated')
    target.clone_docs = copy.deepcopy([docs[2]])
    record = build_keyword_index(source, target, [records[2]])
    assert record['changed_texts'] == 0
    assert record['removed_boilerplate_chunks'] == 0
    assert not target.written
    assert record['expected_target_population_count'] == 10


@pytest.mark.parametrize('mutation', ['analyzer', 'similarity', 'ef_search'])
def test_target_configuration_drift_prevents_completed_record(mutation):
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    source.settings['knn.algo_param.ef_search'] = '100'
    source.settings['similarity'] = {'default': {'type': 'BM25', 'b': '0.75', 'k1': '1.2'}}
    target.clone_docs = copy.deepcopy(docs)
    record = build_keyword_index(source, target, records)
    if mutation == 'analyzer':
        target.mapping['properties']['text']['analyzer'] = 'standard'
    elif mutation == 'similarity':
        target.settings['similarity'] = {'default': {'type': 'BM25', 'b': '0.2', 'k1': '1.2'}}
    else:
        target.settings['knn.algo_param.ef_search'] = '10'
    with pytest.raises(ValueError, match='configuration'):
        verify_keyword_index(source, target, records, record)
    assert record['status'] == 'written_pending_verification'


def test_clone_preserves_source_settings_and_normalizes_representations():
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    source.settings.update({'knn.algo_param.ef_search': '123',
        'similarity': {'default': {'type': 'BM25', 'k1': '1.2'}}, 'refresh_interval': '3s',
        'version': {'created': '136347699'}, 'creation_date': '123456789',
        'provided_name': 'production'})
    target.clone_docs = copy.deepcopy(docs)
    record = build_keyword_index(source, target, records)
    create = next(kwargs['json']['settings']['index'] for method, _, kwargs in target.calls
                  if method == 'PUT')
    assert create['knn.algo_param.ef_search'] == '123'
    assert create['refresh_interval'] == '3s'
    assert not any(name in create for name in ('uuid', 'version', 'creation_date', 'provided_name'))
    # OpenSearch may return scalar settings as strings and nested settings flattened.
    target.settings = {'uuid': 'new-uuid', 'creation_date': 'new-date', 'provided_name': 'isolated',
        'version.created': 'different-creation-version', 'knn': 'true', 'number_of_shards': 1,
        'knn.algo_param.ef_search': 123, 'similarity.default.type': 'BM25',
        'similarity.default.k1': 1.2, 'refresh_interval': '3s'}
    assert verify_keyword_index(source, target, records, record)['status'] == 'complete'



def test_effective_default_setting_drift_is_rejected():
    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    source.defaults = {'knn.algo_param.ef_search': '100'}
    target.defaults = {'knn.algo_param.ef_search': '100'}
    target.clone_docs = copy.deepcopy(docs)
    record = build_keyword_index(source, target, records)
    assert verify_keyword_index(source, target, records, record)['status'] == 'complete'
    target.defaults['knn.algo_param.ef_search'] = '50'
    with pytest.raises(ValueError, match='configuration'):
        verify_keyword_index(source, target, records, record)


@pytest.mark.parametrize('outcome', ['complete', 'task_error', 'interrupted'])
def test_official_async_copy_persists_task_before_bounded_wait(monkeypatch, tmp_path, outcome):
    import json
    import time
    from scripts.benchmark_runtime import atomic_json

    sequence = []
    monkeypatch.setattr(time, 'sleep', lambda seconds: sequence.append(('sleep', seconds)))

    records, docs = fixture()
    source, target = Search('production', docs), Search('isolated')
    target.clone_docs = copy.deepcopy(docs)
    progress_path = tmp_path / '.runtime/keyword-index.json'
    request = target.request
    waits = []
    def asynchronous(method, path, **kwargs):
        if path.startswith('/_reindex?'):
            assert method == 'POST'
            assert path == '/_reindex?wait_for_completion=false&refresh=true'
            return {'task': 'local-node:42'}
        if path.startswith('/_tasks/'):
            assert method == 'GET'
            assert path == '/_tasks/local-node:42'
            assert json.loads(progress_path.read_text())['official_task_id'] == 'local-node:42'
            waits.append(path)
            sequence.append(('query', path))
            if len(waits) == 1:
                return {'completed': False, 'task': {'status': {'created': 5}}}
            if outcome == 'interrupted':
                raise KeyboardInterrupt()
            if outcome == 'task_error':
                return {'completed': True, 'error': {'type': 'private_error',
                        'reason': 'private source text credential'}}
            target.docs = copy.deepcopy(docs)
            return {'completed': True, 'response': {'total': 10, 'created': 10, 'failures': []}}
        return request(method, path, **kwargs)
    target.request = asynchronous
    def progress(record):
        atomic_json(progress_path, record)
    if outcome == 'complete':
        record = build_keyword_index(source, target, records, on_progress=progress)
        assert record['official_task_id'] == 'local-node:42'
        assert record['reindex_created'] == 10
    else:
        expected = KeyboardInterrupt if outcome == 'interrupted' else ValueError
        with pytest.raises(expected):
            build_keyword_index(source, target, records, on_progress=progress)
        assert not target.written
        assert 'private source' not in progress_path.read_text()
    assert len(waits) == 2
    assert sequence == [('query', '/_tasks/local-node:42'), ('sleep', 1),
                        ('query', '/_tasks/local-node:42')]
    assert progress_path.exists()
    assert not any(method == 'DELETE' for method, _, _ in target.calls)
