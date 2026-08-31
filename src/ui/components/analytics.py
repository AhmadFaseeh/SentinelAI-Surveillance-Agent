"""
SentinelAI Component: Movement & Segment Analytics
"""

import streamlit as st
from typing import List
from src.core.types import VideoSegment
from src.ingestion.video_reader import VideoReader


def render_activity_timeline(segments: List[VideoSegment]):
    """Renders visual breakdown of active vs idle temporal segments."""
    if not segments:
        st.info("No activity segments indexed yet.")
        return

    st.markdown("### Temporal Activity Segments")
    for seg in segments:
        color = "#00e676" if seg.activity_level == "high" else "#ffab00"
        st.markdown(f"""
        <div style="background: rgba(22, 30, 46, 0.7); border-left: 4px solid {color}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <b>Segment {seg.segment_id}:</b> {VideoReader.format_time(seg.start_sec)} - {VideoReader.format_time(seg.end_sec)}
                <div style="color: #8b949e; font-size: 12px; margin-top: 2px;">{seg.summary or 'Dynamic activity recorded'}</div>
            </div>
            <span style="background: rgba(255, 255, 255, 0.08); padding: 4px 10px; border-radius: 12px; font-size: 11px; text-transform: uppercase;">
                {seg.activity_level}
            </span>
        </div>
        """, unsafe_allow_html=True)
