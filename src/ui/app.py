"""
SentinelAI: Autonomous Surveillance & CCTV Video Forensic Dashboard
Main Application Entrypoint with Clean Rectangular Video Alignment and Minimalist Dark Design.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path

from src.core.config import default_config
from src.core.logger import logger
from src.ingestion.video_reader import VideoReader
from src.ingestion.scene_segmenter import SceneSegmenter
from src.ingestion.live_camera import LiveCameraRecorder
from src.perception.detector import SurveillanceDetector
from src.perception.tracker import SurveillanceTracker
from src.perception.action_detector import ActionDetector
from src.embeddings.siglip_embedder import MultimodalEmbedder
from src.embeddings.vector_store import VideoVectorStore
from src.agent.tools import SurveillanceTools
from src.agent.graph import ForensicAgent
from src.llm.local_llm import LocalLLMClient

# UI Styles & Components
from src.ui.styles import CUSTOM_CSS
from src.ui.components import (
    render_header,
    render_evidence_card,
    render_activity_timeline,
    render_report_view
)

st.set_page_config(
    page_title="SentinelAI - Operations Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling tokens
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_core_models():
    """Initializes models once to conserve VRAM."""
    embedder = MultimodalEmbedder()
    detector = SurveillanceDetector()
    tracker = SurveillanceTracker()
    action_detector = ActionDetector(detector=detector)
    vector_store = VideoVectorStore()
    llm_client = LocalLLMClient()
    return embedder, detector, tracker, action_detector, vector_store, llm_client


def main():
    # 1. Top HUD Header
    render_header()

    embedder, detector, tracker, action_detector, vector_store, llm_client = load_core_models()

    # 2. Input Source Selector (Clean Border Container)
    with st.container(border=True):
        st.markdown("### Step 1: Select Surveillance Input Source")
        source_mode = st.radio(
            "Input Mode:",
            ["Live Camera / Webcam (Real-time Action Recognition)", "Upload Video File (CCTV / Surveillance MP4)"],
            index=0,
            horizontal=True
        )

    # ----------------- Mode 1: Live Camera Feed -----------------
    if "Live Camera" in source_mode:
        with st.container(border=True):
            st.markdown("### Live Camera & Posture Recorder")
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1.5])
            with col_c1:
                cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, help="0 is default webcam")
            with col_c2:
                rec_duration = st.slider("Record Duration (seconds)", min_value=5, max_value=30, value=10, step=5)
            with col_c3:
                st.write("")
                st.write("")
                record_btn = st.button("Start Live Recording & Auto-Analyze", type="primary", use_container_width=True)

            if record_btn:
                with st.status("Recording from Live Webcam...", expanded=True) as status:
                    st.write("Connecting to camera device...")
                    cam = LiveCameraRecorder(camera_index=cam_idx, fps=15)
                    if not cam.start_camera():
                        st.error("Could not access camera! Please check your webcam connection.")
                    else:
                        out_path = f"./data/live_recordings/clip_{int(time.time())}.mp4"
                        cam.start_recording(out_path)

                        live_preview = st.empty()
                        progress_bar = st.progress(0.0)
                        start_t = time.time()

                        for step in range(int(rec_duration * 15)):
                            elapsed = time.time() - start_t
                            if elapsed >= rec_duration:
                                break

                            frame = cam.get_frame()
                            if frame is not None:
                                annotated, action_info = action_detector.analyze_live_frame(frame)
                                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                live_preview.image(rgb, caption=f"Live HUD: {action_info['status_text']} | Frame #{cam.frames_count}", channels="RGB")

                            progress_bar.progress(min(1.0, elapsed / rec_duration))
                            time.sleep(0.04)

                        cam.stop_recording()
                        cam.release()

                        if cam.frames_count == 0 or not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
                            status.update(label="Recording Failed!", state="error")
                            st.error("No frames were captured from camera. Please check camera index.")
                        else:
                            st.session_state["video_path"] = out_path
                            st.session_state["video_name"] = Path(out_path).name
                            st.session_state["indexed"] = False

                            st.write("Indexing recorded clip with SigLIP & Motion Filter...")
                            snapshots_dir = Path("./data/snapshots") / Path(out_path).stem
                            segmenter = SceneSegmenter()
                            keyframes, segments = segmenter.process_video(
                                out_path,
                                output_snapshots_dir=str(snapshots_dir)
                            )

                            images_for_embed = []
                            reader = VideoReader(out_path)
                            for kf in keyframes:
                                f = reader.get_frame_at_timestamp(kf.timestamp_sec)
                                if f is not None:
                                    boxes = detector.detect_in_frame(f)
                                    kf.detected_objects = list(set([b.class_name for b in boxes]))
                                    images_for_embed.append(VideoReader.bgr_to_pil(f))

                            if images_for_embed:
                                embeddings = embedder.embed_image_batch(images_for_embed)
                                vector_store.clear()
                                vector_store.add_keyframes(
                                    keyframes=keyframes,
                                    embeddings=embeddings,
                                    video_id=Path(out_path).stem
                                )

                            st.session_state["keyframes"] = keyframes
                            st.session_state["segments"] = segments
                            st.session_state["indexed"] = True
                            status.update(label="Recorded & Indexed Successfully!", state="complete")
                            st.success(f"Captured {cam.frames_count} frames ({rec_duration}s) and indexed {len(keyframes)} active keyframes!")

    # ----------------- Mode 2: Upload Video File -----------------
    else:
        with st.container(border=True):
            st.markdown("### Upload Surveillance Footage")
            uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_file:
                if "video_name" not in st.session_state or st.session_state["video_name"] != uploaded_file.name:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    st.session_state["video_path"] = tfile.name
                    st.session_state["video_name"] = uploaded_file.name
                    st.session_state["indexed"] = False

            if "video_path" in st.session_state:
                st.success(f"Loaded: {st.session_state.get('video_name', 'Video')}")
                if st.button("Process & Index Video", type="primary"):
                    with st.spinner("Processing video (Motion filtering, Object tracking, Vector embeddings)..."):
                        snapshots_dir = Path("./data/snapshots") / Path(st.session_state["video_name"]).stem
                        segmenter = SceneSegmenter()
                        keyframes, segments = segmenter.process_video(
                            st.session_state["video_path"],
                            output_snapshots_dir=str(snapshots_dir)
                        )

                        images_for_embed = []
                        reader = VideoReader(st.session_state["video_path"])
                        for kf in keyframes:
                            frame = reader.get_frame_at_timestamp(kf.timestamp_sec)
                            if frame is not None:
                                boxes = detector.detect_in_frame(frame)
                                kf.detected_objects = list(set([b.class_name for b in boxes]))
                                images_for_embed.append(VideoReader.bgr_to_pil(frame))

                        if images_for_embed:
                            embeddings = embedder.embed_image_batch(images_for_embed)
                            vector_store.clear()
                            vector_store.add_keyframes(
                                keyframes=keyframes,
                                embeddings=embeddings,
                                video_id=Path(st.session_state["video_name"]).stem
                            )

                        st.session_state["keyframes"] = keyframes
                        st.session_state["segments"] = segments
                        st.session_state["indexed"] = True
                        st.success(f"Indexed {len(keyframes)} active keyframes!")

    # 3. Telemetry & Video Player Grid (60% Video, 40% Telemetry)
    if "video_path" in st.session_state and st.session_state["video_path"]:
        with st.container(border=True):
            col1, col2 = st.columns([1.5, 1])

            with col1:
                st.markdown("### Surveillance Playback")
                st.video(st.session_state["video_path"])

            with col2:
                st.markdown("### Session Telemetry")
                with VideoReader(st.session_state["video_path"]) as vr:
                    meta = vr.get_metadata()
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("DURATION", meta["duration_formatted"])
                    mcol2.metric("FPS", f"{meta['fps']:.1f}")

                    mcol3, mcol4 = st.columns(2)
                    mcol3.metric("RESOLUTION", f"{meta['width']}x{meta['height']}")
                    mcol4.metric("INDEXED FRAMES", len(st.session_state.get("keyframes", [])))
                    
                    st.info(f"Active Feed: `{st.session_state.get('video_name', 'Footage')}`")

        # 4. Forensic Investigation Tabs
        tab1, tab2, tab3 = st.tabs([
            "AI Detective Agent (Forensic Search)",
            "Activity & Movement Segments",
            "Forensic Incident Report"
        ])

        with tab1:
            st.markdown("### Natural Language Forensic Search")
            st.caption("Ask queries in plain English (e.g. 'Person standing up', 'Holding mobile phone', 'White car with plate BUS 268').")

            qcol1, qcol2 = st.columns([3, 1])
            with qcol1:
                query = st.text_input("Enter Investigation Query:", placeholder="e.g. Find white car or person sitting")
            with qcol2:
                st.write("")
                st.write("")
                run_investigation_btn = st.button("Investigate", type="primary", use_container_width=True)

            if run_investigation_btn and query:
                if not st.session_state.get("indexed", False):
                    st.warning("Please index the footage first!")
                else:
                    with st.spinner("AI Detective Agent analyzing surveillance timeline..."):
                        reader = VideoReader(st.session_state["video_path"])
                        tools = SurveillanceTools(
                            embedder=embedder,
                            vector_store=vector_store,
                            tracker=tracker,
                            video_reader=reader
                        )
                        agent = ForensicAgent(tools=tools, llm_client=llm_client)
                        report = agent.run_investigation(query=query, video_source=st.session_state.get("video_name", "surveillance"))
                        st.session_state["last_report"] = report

            if "last_report" in st.session_state:
                rep = st.session_state["last_report"]
                st.markdown(f"""
                <div class="verdict-box-confirmed">
                    <h4 style="margin:0 0 6px 0; color:#00e676;">Case Verdict: {rep.verdict}</h4>
                    <p style="margin:0; font-size:13px; color:#e6edf3;">{rep.summary_findings}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### Key Evidence Discovered")
                for idx, ev in enumerate(rep.evidence_items, 1):
                    render_evidence_card(ev, index=idx)

        with tab2:
            render_activity_timeline(st.session_state.get("segments", []))

        with tab3:
            if "last_report" in st.session_state:
                render_report_view(st.session_state["last_report"])
            else:
                st.info("Run an investigation query in Tab 1 to generate an incident report.")


if __name__ == "__main__":
    main()
