"""
SentinelAI: LangGraph Agent State Definition
"""

from typing import TypedDict, List, Dict, Any, Optional
from ..core.types import ForensicEvidence, IncidentReport


class AgentInvestigationState(TypedDict):
    """LangGraph state representation for the CCTV Detective Agent."""
    query: str
    video_source: str
    step_count: int
    max_steps: int
    is_finished: bool
    plan: List[str]
    current_thought: str
    next_tool: Optional[str]
    tool_input: Dict[str, Any]
    tool_outputs: List[Dict[str, Any]]
    candidate_timestamps: List[float]
    evidence_collected: List[ForensicEvidence]
    final_report: Optional[IncidentReport]
