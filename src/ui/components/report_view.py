"""
SentinelAI Component: Forensic Case Report Exporter
"""

import streamlit as st
import json
from src.core.types import IncidentReport


def render_report_view(report: IncidentReport):
    """Renders structured case verdict, executive summary, and export actions."""
    verdict_color = "#00e676" if report.verdict == "CONFIRMED_EVENT" else "#ffab00" if report.verdict == "INCONCLUSIVE" else "#ff5252"

    st.markdown(f"""
    <div style="background: rgba(22, 30, 46, 0.85); border: 1px solid {verdict_color}; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; color: #ffffff; font-size: 16px;">Case Report: {report.case_title}</h3>
            <span style="background: {verdict_color}22; color: {verdict_color}; border: 1px solid {verdict_color}; padding: 3px 10px; border-radius: 16px; font-weight: 700; font-size: 11px; font-family: monospace;">
                {report.verdict}
            </span>
        </div>
        <div style="color: #8b949e; font-size: 12px; margin-bottom: 8px;">
            <b>Source:</b> <code>{report.video_source}</code> | <b>Query:</b> <i>"{report.query}"</i>
        </div>
        <div style="color: #e6edf3; font-size: 14px; line-height: 1.6; background: rgba(0, 0, 0, 0.2); padding: 10px; border-radius: 6px;">
            {report.summary_findings}
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_dict = report.model_dump()
    json_data = json.dumps(report_dict, indent=2)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "Download JSON Incident Report",
            data=json_data,
            file_name="sentinel_incident_report.json",
            mime="application/json",
            use_container_width=True
        )
    with col2:
        markdown_summary = f"""# Forensic Incident Report: {report.case_title}
- **Verdict:** {report.verdict}
- **Source:** {report.video_source}
- **Query:** {report.query}
- **Summary:** {report.summary_findings}

## Evidence Items ({len(report.evidence_items)})
""" + "\n".join([f"- **[{ev.timestamp_formatted}]** {ev.description} (Confidence: {ev.confidence:.2f})" for ev in report.evidence_items])
        st.download_button(
            "Download Markdown Case Summary",
            data=markdown_summary,
            file_name="sentinel_incident_report.md",
            mime="text/markdown",
            use_container_width=True
        )
