"""Bind assistant source-review observations to the exact stored cited chunks.

No review labels are inferred by this script. It adds content hashes, never article
bodies, so later readers can detect source/output drift before trusting the review.
"""
import hashlib
import json
from pathlib import Path

from multihop_outputs import dataset, digest, evidence_text_by_chunk, payloads

PATH = Path('artifacts/verdict/source-review-r3.json')


def main():
    review = json.loads(PATH.read_text())
    questions = {f'MH-{key[:12]}': row for key, row in dataset().items()}
    outputs = payloads()
    for row in review['rows']:
        question = questions[row['item_id']]
        payload = outputs[row['arm']][digest(question['query'])]
        claims = payload.get('claims', [])
        ids = sorted({ref for claim in claims for ref in claim['evidence_ids']})
        chunks = evidence_text_by_chunk(ids)
        if set(chunks) != set(ids):
            raise ValueError(f"Missing cited source for {row['item_id']}/{row['arm']}")
        row['query_sha256'] = digest(question['query'])
        row['claims_sha256'] = digest(json.dumps(claims, ensure_ascii=False, sort_keys=True))
        row['cited_chunk_sha256'] = {key: digest(chunks[key]) for key in ids}
    review['input_sha256'] = hashlib.sha256(Path(review['input']).read_bytes()).hexdigest()
    PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
    print(f"Fingerprinted {len(review['rows'])} assistant reviews; human_reviewed remains {review['human_reviewed']}")


if __name__ == '__main__':
    main()
