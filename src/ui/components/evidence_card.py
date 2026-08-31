"""
SentinelAI Component: Forensic Evidence Card
"""

import streamlit as st
from pathlib import Path
from src.core.types import ForensicEvidence


def render_evidence_card(evidence: ForensicEvidence, index: int = 1):
    """Renders a structured glassmorphic card for an individual forensic discovery."""
    score_pct = int(evidence.confidence * 100)
    
    st.markdown(f"""
    <div class="evidence-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div>
                <span class="evidence-timestamp">{evidence.timestamp_formatted}</span>
                <span style="color: #8b949e; font-size: 12px; margin-left: 10px;">({evidence.evidence_type})</span>
            </div>
            <span class="evidence-score">{score_pct}% Match</span>
        </div>
        <div style="color: #c9d1d9; font-size: 13px; line-height: 1.5;">
            {evidence.description}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if evidence.snapshot_path and Path(evidence.snapshot_path).exists():
        st.image(
            evidence.snapshot_path,
            width=360,
            caption=f"Snapshot @ {evidence.timestamp_formatted} (Match: {score_pct}%)"
        )
