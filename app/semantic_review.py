"""Versioned source-entailment rubric, used for offline/shadow review only."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RUBRIC_VERSION = 'source-entailment-v4'
RUBRIC = '''Judge the claim against only the supplied evidence. Do not use outside knowledge.
Return entailment=supported only if EVERY substantive assertion follows from evidence.
partial means some assertions are supported and others are missing; unsupported means
an atomic assertion is unsupported or contradicted. unclear is reserved for ambiguous
input, not a substitute for an identifiable mistake. Rephrasing is allowed.
Check these independently:
1. Subject/source: the speaker is substantive. A publisher reporting a lawyer's argument
is not endorsing it. Preserve who said/believed/alleged what. Reporting verbs such as
attributed to, according to and said are equivalent if attribution is preserved.
2. Modality: may/might/could, opinion, forecast and allegation are NOT confirmed fact.
Equivalent uncertainty words are accepted; do not require one fixed synonym.
3. Negation scope: missing from excerpts cannot prove never mentioned anywhere in a
whole article. That atomic global absence is unsupported, not partial. An explicitly
quoted denial of a requirement is an ordinary supported negative fact.
4. Attribute: compare only the asked attribute. Equal launch dates and unequal hours
support date consistency. Unequal availability does not disprove strategy consistency.
Missing one side means insufficient evidence, not contradiction. Subject and period
must align, and all requested sources must be covered.
5. Time: publication date, event date and effective date are different. A policy's value
is applicable only inside its effective interval; version creation time is not its
legal effective date. Dates/source headers are substantive when the question asks them.
6. Conclusion: supporting individual source facts does not automatically support a
comparison or a yes/no conclusion. Inspect the relation actually asserted. A correct
single-side fact can be supported but relevance=related rather than answers.
For each dimension use supported/error/not_applicable/unclear. Return short reasons,
not chain-of-thought. Do not see a reference answer or a prior judge's label.'''


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entailment: Literal['supported','partial','unsupported','unclear']
    relevance: Literal['answers','related','off_topic','unclear']
    subject: Literal['supported','error','not_applicable','unclear']
    modality: Literal['supported','error','not_applicable','unclear']
    negation_scope: Literal['supported','error','not_applicable','unclear']
    attribute: Literal['supported','error','not_applicable','unclear']
    time: Literal['supported','error','not_applicable','unclear']
    conclusion: Literal['supported','error','not_applicable','unclear']
    reason: str = Field(max_length=220)


def review(models, question, claim, evidence):
    from app.config import settings
    result = models._post('/api/chat', {
        'model':settings().chat_model, 'stream':False, 'keep_alive':'30m',
        'format':ReviewResult.model_json_schema(),
        'messages':[{'role':'system','content':RUBRIC},
                    {'role':'user','content':__import__('json').dumps(
                        {'question':question,'claim':claim,'evidence':evidence},ensure_ascii=False)}],
        'options':{'temperature':0,'seed':42,'num_ctx':8192,'num_predict':230},
    })
    return ReviewResult.model_validate_json(result['message']['content']).model_dump(), {
        'prompt_tokens':result.get('prompt_eval_count',0),
        'completion_tokens':result.get('eval_count',0),
    }


class CompactReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    entailment: Literal['supported', 'partial', 'unsupported', 'unclear']
    relevance: Literal['answers', 'related', 'off_topic', 'unclear']
    failure_type: Literal['none', 'subject', 'modality', 'negation_scope', 'attribute', 'time', 'conclusion', 'mixed']
    reason: str = Field(max_length=160)


def compact_review(models, question, claim, evidence):
    """A separate, cheaper candidate rubric; never relabel stored v4 results."""
    import json
    from app.config import settings
    result = models._chat({
        'model': settings().chat_model, 'stream': False, 'keep_alive': '30m',
        'format': CompactReview.model_json_schema(),
        'messages': [{'role': 'system', 'content': (
            'Evaluate evidence support and question relevance SEPARATELY. No outside knowledge. '
            'First read exactly what the claim asserts, not what the question asks. '
            'supported: every asserted fact and relation follows from evidence. '
            'unsupported: a single atomic assertion is false or lacks evidence. '
            'partial: a genuinely multi-assertion claim mixes supported and unsupported assertions. '
            'unclear: the input itself is ambiguous. A source reporting a speaker is not endorsing them. '
            'may/might/could are equivalent uncertainty. Excerpt absence is not whole-document absence. '
            'Explicit source denials are supported negative facts. Compare ONLY the attribute in the claim: '
            'different hours do not contradict equal opening dates; availability is not strategy. '
            'Publication/creation dates are not effective dates. Correct isolated facts do not prove a relation. '
            'Relevance answers means all asked items and the final relation are answered; related means '
            'relevant facts but an asked relation/item is missing. Return short JSON, no reasoning trace.'
        )}, {'role': 'user', 'content': json.dumps(
            {'question': question, 'claim': claim, 'evidence': evidence}, ensure_ascii=False)}],
        'options': {'temperature': 0, 'seed': 42, 'num_ctx': 4096, 'num_predict': 150},
    })
    return CompactReview.model_validate_json(result['message']['content']).model_dump(), {
        'prompt_tokens': result.get('prompt_eval_count', 0),
        'completion_tokens': result.get('eval_count', 0)}
