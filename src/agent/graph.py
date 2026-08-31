"""
SentinelAI: LangGraph Autonomous Forensic Agent
Orchestrates multi-step video reasoning, tool execution, OCR plate reading, and evidence compilation.
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END

from .state import AgentInvestigationState
from .tools import SurveillanceTools
from ..llm.local_llm import LocalLLMClient
from ..core.types import ForensicEvidence, IncidentReport
from ..core.logger import logger


class ForensicAgent:
    def __init__(self, tools: SurveillanceTools, llm_client: LocalLLMClient):
        self.tools = tools
        self.llm = llm_client
        self.graph = self._build_graph()

    def _planner_node(self, state: AgentInvestigationState) -> Dict[str, Any]:
        """Reasons over query and selects appropriate tool or decides to conclude."""
        step = state["step_count"] + 1
        query = state["query"]

        logger.info(f"[bold cyan]Detective Agent Step {step}:[/bold cyan] Analyzing query: '{query}'")

        # Step 1: Initial visual search if no evidence collected yet
        if len(state["evidence_collected"]) == 0:
            return {
                "step_count": step,
                "current_thought": f"Searching video database for visual cues matching: '{query}'",
                "next_tool": "search_visual_scenes",
                "tool_input": {"query": query, "top_k": 5}
            }

        # Conclude if evidence is gathered or step limit reached
        return {
            "step_count": step,
            "current_thought": "Sufficient evidence collected. Synthesizing incident report.",
            "next_tool": "conclude",
            "is_finished": True
        }

    def _tool_executor_node(self, state: AgentInvestigationState) -> Dict[str, Any]:
        """Executes the chosen tool."""
        tool_name = state["next_tool"]
        tool_input = state["tool_input"]
        query_lower = state["query"].lower()

        outputs = list(state["tool_outputs"])
        new_evidence = list(state["evidence_collected"])
        timestamps = list(state["candidate_timestamps"])

        if tool_name == "search_visual_scenes":
            results = self.tools.search_visual_scenes(
                query=tool_input.get("query", state["query"]),
                top_k=tool_input.get("top_k", 5)
            )
            outputs.append({"tool": tool_name, "results": results})

            is_plate_query = any(k in query_lower for k in ["plate", "number plate", "number", "text", "read"])

            for res in results:
                t_sec = res["timestamp_sec"]
                timestamps.append(t_sec)

                desc = f"Visual similarity match (Score: {res['similarity_score']:.2f}). {res.get('document', '')}"
                
                # If user asked about number plate / text, run OCR on candidate frame
                if is_plate_query:
                    ocr_texts = self.tools.read_license_plate_ocr(t_sec)
                    if ocr_texts:
                        desc += f" | 🔍 Extracted Plate/Text: {', '.join(ocr_texts)}"

                ev = ForensicEvidence(
                    timestamp_sec=t_sec,
                    timestamp_formatted=res["timestamp_formatted"],
                    evidence_type="ocr_plate" if is_plate_query else "visual_match",
                    description=desc,
                    confidence=res["similarity_score"],
                    snapshot_path=res.get("frame_path")
                )
                new_evidence.append(ev)

        elif tool_name == "get_track_history":
            track_id = tool_input.get("track_id", 0)
            res = self.tools.get_track_history(track_id)
            outputs.append({"tool": tool_name, "results": res})

        return {
            "tool_outputs": outputs,
            "evidence_collected": new_evidence,
            "candidate_timestamps": sorted(list(set(timestamps)))
        }

    def _reporter_node(self, state: AgentInvestigationState) -> Dict[str, Any]:
        """Synthesizes the collected evidence into an IncidentReport."""
        evidence = state["evidence_collected"]
        query = state["query"]
        video_src = state["video_source"]

        if len(evidence) > 0:
            top_match = evidence[0]
            verdict = "CONFIRMED_EVENT"
            summary = (
                f"Incident analysis for query '{query}' identified {len(evidence)} relevant keyframe events. "
                f"Primary activity identified at timestamp {top_match.timestamp_formatted} "
                f"with confidence score {top_match.confidence:.2f}."
            )
        else:
            verdict = "NO_MATCH"
            summary = f"No matching surveillance events found in footage for query '{query}'."

        report = IncidentReport(
            case_title=f"Forensic Case: {query[:40]}...",
            video_source=video_src,
            query=query,
            summary_findings=summary,
            evidence_items=evidence,
            timeline_timestamps=[e.timestamp_formatted for e in evidence],
            verdict=verdict
        )

        logger.info(f"[bold green]Case Closed:[/bold green] Verdict = {verdict}")
        return {"final_report": report, "is_finished": True}

    def _router(self, state: AgentInvestigationState) -> str:
        if state.get("is_finished") or state["next_tool"] == "conclude":
            return "reporter"
        return "tool_executor"

    def _build_graph(self):
        builder = StateGraph(AgentInvestigationState)

        builder.add_node("planner", self._planner_node)
        builder.add_node("tool_executor", self._tool_executor_node)
        builder.add_node("reporter", self._reporter_node)

        builder.set_entry_point("planner")
        builder.add_conditional_edges("planner", self._router, {
            "tool_executor": "tool_executor",
            "reporter": "reporter"
        })
        builder.add_edge("tool_executor", "planner")
        builder.add_edge("reporter", END)

        return builder.compile()

    def run_investigation(self, query: str, video_source: str = "surveillance_feed") -> IncidentReport:
        """Runs the complete investigative cycle and returns a structured IncidentReport."""
        initial_state: AgentInvestigationState = {
            "query": query,
            "video_source": video_source,
            "step_count": 0,
            "max_steps": 5,
            "is_finished": False,
            "plan": [],
            "current_thought": "Initializing investigation...",
            "next_tool": None,
            "tool_input": {},
            "tool_outputs": [],
            "candidate_timestamps": [],
            "evidence_collected": [],
            "final_report": None
        }

        final_state = self.graph.invoke(initial_state)
        return final_state["final_report"]
