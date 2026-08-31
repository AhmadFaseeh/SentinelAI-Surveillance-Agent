import streamlit as st
def render_header():
    st.markdown("""
    <div class="hud-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1 class="hud-title">SentinelAI Operations Center</h1>
                <p style="color: #8b949e; margin: 4px 0 0 0; font-size: 13px;">
                    Autonomous Surveillance & Forensic Investigation Engine
                </p>
            </div>
            <div style="margin-top: 6px;">
                <span class="hud-badge badge-online">SYSTEM ONLINE</span>
                <span class="hud-badge badge-vram">8GB VRAM OPTIMIZED</span>
                <span class="hud-badge badge-local">100% LOCAL & OFFLINE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
