#  SentinelAI: Autonomous Local CCTV & Surveillance Video Forensic Agent

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/VRAM-8GB%20Optimized-green.svg)]()
[![Model](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Multimodal](https://img.shields.io/badge/Embeddings-SigLIP%20%2F%20CLIP-purple.svg)](https://github.com/mlfoundations/open_clip)
[![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-red.svg)](https://www.trychroma.com/)
[![Agentic](https://img.shields.io/badge/Orchestrator-LangGraph-yellow.svg)](https://github.com/langchain-ai/langgraph)

**SentinelAI** is an end-to-end, 100% offline, and privacy-preserving AI agent built for CCTV and surveillance footage forensic investigation. It requires **zero paid APIs (No OpenAI/Claude)** and runs comfortably within an **8 GB VRAM** consumer GPU footprint.

---

##  System Architecture

```
[ CCTV / Surveillance Video (.mp4 / .avi / RTSP) ]
                        │
                        ▼
       [ 1. Ingestion & Motion Energy Filter ]
         - OpenCV MOG2 Background Subtractor
         - Static Frame Elimination (90% compression)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
[ 2. Perception Engine ]      [ 3. Multimodal Indexing ]
  - YOLOv8s Detection           - Google SigLIP (ViT-B/16)
  - ByteTrack Trajectories      - ChromaDB Vector Store
         │                             │
         └──────────────┬──────────────┘
                        ▼
       [ 4. LangGraph Forensic Detective Brain ]
         - Local Qwen 2.5 (3B / 7B) Controller
         - ReAct State Machine (Search -> Track -> Inspect -> Report)
                        │
                        ▼
         [ 5. Security Officer Dashboard ]
           - Streamlit Web UI with Time-Seeking Video Player
           - Automated Forensic Incident Report Exporter (JSON)
```

---

##  Key Features

1. **Intelligent Ingestion & Inactivity Pruning:**
   * Eliminates dead CCTV footage using MOG2 background subtraction.
   * Samples dynamically during bursts of motion energy.

2. **Multimodal Forensic Search (Video RAG):**
   * Search through hours of surveillance with natural language (e.g. *"Show when the blue car entered the gate"*, *"Person in black carrying a backpack"*).

3. **Multi-Object Tracking & Dwell-Time Analysis:**
   * Tracks entity movement paths and alerts on loitering / suspicious dwelling (>60s).

4. **Autonomous Investigative Agent:**
   * Uses LangGraph to plan multi-step inquiries, inspect visual evidence, and generate structured forensic case verdicts (`CONFIRMED_EVENT`, `INCONCLUSIVE`, `NO_MATCH`).

5. **100% Local & Privacy Guaranteed:**
   * Zero cloud dependencies. All weights and vectors remain on the local machine.

---

##  Quick Start Guide

### 1. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (already completed in workspace)
pip install -r requirements.txt
```

### 2. Run End-to-End CLI Demo
```bash
python demo_pipeline.py
```

### 3. Launch Interactive Security Dashboard
```bash
streamlit run src/ui/app.py
```

---

##  Project Structure

```
├── configs/                  # Hyperparameters and threshold settings
├── data/                     # Local storage for videos, snapshots, and vector DBs
├── src/
│   ├── core/                 # Configuration, schemas (Pydantic), and rich logger
│   ├── ingestion/            # Video reader, MOG2 motion detector, scene segmenter
│   ├── perception/           # YOLOv8 detector & ByteTrack multi-object tracker
│   ├── embeddings/           # SigLIP multimodal embedder & ChromaDB vector store
│   ├── agent/                # LangGraph state, tools, and reasoning graph
│   ├── llm/                  # Local Ollama / offline LLM interface
│   └── ui/                   # Streamlit web dashboard
├── tests/                    # Pytest test suite
├── demo_pipeline.py          # Standalone end-to-end CLI demonstration
└── requirements.txt          # Python dependencies
```

---

##  Academic & FYP Highlights
* **Core Contribution:** Hybrid spatio-temporal video indexing combining Background Subtraction with Multimodal Contrastive Embeddings (SigLIP) and Cyclic Agentic Graphs.
* **Evaluation Metrics:** Motion compression ratio, Recall@K retrieval accuracy, FPS throughput on 8GB VRAM.
