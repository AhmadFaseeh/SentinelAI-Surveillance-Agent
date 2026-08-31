"""
SentinelAI: Modern Cyber-Security UI Design System
Clean Minimalist Dark Theme without emojis, with 16:9 Rectangular Video Framing.
"""

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Background & Base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #080a0f 95%) !important;
        color: #e6edf3 !important;
    }

    /* Top HUD Header Banner */
    .hud-banner {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95) 0%, rgba(22, 33, 62, 0.8) 100%);
        border: 1px solid rgba(0, 229, 255, 0.35);
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.08);
    }

    .hud-title {
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00e5ff 0%, #00e676 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-transform: uppercase;
    }

    .hud-badge {
        display: inline-block;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 16px;
        margin-left: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-online {
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.4);
    }

    .badge-vram {
        background: rgba(0, 229, 255, 0.15);
        color: #00e5ff;
        border: 1px solid rgba(0, 229, 255, 0.4);
    }

    .badge-local {
        background: rgba(255, 171, 0, 0.15);
        color: #ffab00;
        border: 1px solid rgba(255, 171, 0, 0.4);
    }

    /* Fixed Rectangular (16:9 ratio) Video Player Framing */
    div[data-testid="stVideo"] {
        width: 100% !important;
        max-height: 270px !important;
        height: 270px !important;
        background: #000000 !important;
        border: 1px solid rgba(0, 229, 255, 0.25) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stVideo"] video {
        width: 100% !important;
        height: 100% !important;
        max-height: 270px !important;
        object-fit: cover !important;
        border-radius: 8px !important;
        margin: 0 auto !important;
        display: block !important;
    }

    /* Native Container Border Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(18, 24, 38, 0.75) !important;
        border: 1px solid rgba(0, 229, 255, 0.18) !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(22, 30, 46, 0.85) !important;
        border: 1px solid rgba(0, 229, 255, 0.22) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 19px !important;
        color: #00e5ff !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        color: #8b949e !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Evidence Card */
    .evidence-card {
        background: linear-gradient(90deg, rgba(22, 30, 46, 0.9) 0%, rgba(18, 24, 38, 0.7) 100%);
        border-left: 4px solid #00e676;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 10px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .evidence-card:hover {
        transform: translateY(-2px);
        border-left-color: #00e5ff;
    }

    .evidence-timestamp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        color: #00e5ff;
    }

    .evidence-score {
        float: right;
        background: rgba(0, 230, 118, 0.15);
        color: #00e676;
        padding: 2px 7px;
        border-radius: 10px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* Verdict Alert Box */
    .verdict-box-confirmed {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.12) 0%, rgba(0, 77, 64, 0.25) 100%);
        border: 1px solid rgba(0, 230, 118, 0.4);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton>button {
        font-weight: 700 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
        font-weight: 600;
        background: rgba(22, 30, 46, 0.5);
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 229, 255, 0.15) !important;
        border-bottom: 2px solid #00e5ff !important;
        color: #00e5ff !important;
    }
</style>
"""
