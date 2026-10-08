"""Conservative cost routing. Explicit dynamic requests remain explicit.

Free-form dynamic planning is an experimental control only: on frozen Dev it matched
single RAG at 2.4x latency and lost to the fixed workflow on Hard. Questions whose
next step depends on an observation go to the bounded Hybrid agent instead.
"""
import re

from app.task_analysis import conflict_intent, historical_route_intent, multi_source_intent
from agent.planner import subgoals


def execution_route(question, *, document_id=None):
    dependent = bool(re.search(
        r'(?:如果|只有|根据(?:查询|检索|第一步|上述|该).{0,20}(?:结果|线索|标识))'
        r'|\b(?:depending on|based on (?:the )?(?:first|retrieved|search)'
        r'.{0,25}(?:result|finding)|if .{1,60} then)\b', question, re.I))
    if document_id or historical_route_intent(question):
        return {'mode': 'workflow', 'reason': 'explicit_version_workflow'}
    if dependent:
        return {'mode': 'hybrid', 'reason': 'observation_dependent_bounded_agent'}
    independent_asks = bool(re.search(r'\band\s+(?:what|which|who|when|where|why|how)\b', question, re.I))
    explicit_source_group = bool(re.search(
        r'\b(?:two|three|four|both|all)\s+(?:notices?|documents?|reports?|articles?|sources?)\b',
        question, re.I))
    if (len(subgoals(question)) > 1 or independent_asks or explicit_source_group
            or multi_source_intent(question) or conflict_intent(question)):
        return {'mode': 'workflow', 'reason': 'fixed_compound_workflow'}
    return {'mode': 'workflow', 'reason': 'direct_retrieval', 'direct': True}
