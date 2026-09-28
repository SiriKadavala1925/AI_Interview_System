# 🎙️ AI Interview Simulation System

An autonomous, multimodal technical interview simulation and assessment platform powered by local Small Language Models (SLMs) and vision models. Practice real-world technical interviews for top tech companies with voice transcription, live whiteboard diagramming, intelligent proctoring, and comprehensive rubric-calibrated evaluation—all running 100% locally and privately without paid external APIs.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture & Tech Stack](#-architecture--tech-stack)
- [System Requirements](#-system-requirements)
- [Installation & Setup](#-installation--setup)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Install Local SLMs via Ollama](#2-install-local-slms-via-ollama)
  - [3. Install FFmpeg](#3-install-ffmpeg)
  - [4. Set Up Python Virtual Environment](#4-set-up-python-virtual-environment)
  - [5. Initialize the SQLite Database](#5-initialize-the-sqlite-database)
- [Running the Application](#-running-the-application)
- [Interview Workflow](#-interview-workflow)
- [Evaluation & Scoring Rubric](#-evaluation--scoring-rubric)
- [Directory Structure](#-directory-structure)
- [Testing & Verification Utilities](#-testing--verification-utilities)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🌟 Overview

The **AI Interview Simulation System** simulates high-stakes technical interviews calibrated to industry standards at tier-1 technology firms (Google, Microsoft, Amazon, Meta, Apple, etc.). Unlike conventional interview bots that depend on closed-source cloud APIs, this system utilizes privacy-first, on-device SLMs:

- **Reasoning & Evaluation**: Microsoft **Phi-3** via Ollama.
- **Visual Whiteboard Analysis**: **Moondream2** vision model via Ollama.
- **Voice Transcription**: **Faster-Whisper** (`base` int8 CPU) for local, low-latency Speech-to-Text.
- **Question Delivery**: **pyttsx3** for offline Text-to-Speech question narration.
- **Integrity & Proctoring**: Haar-Cascade face detection via OpenCV and WebRTC background monitoring.

---

## ✨ Key Features

### 1. 🎯 Role & Company-Tailored Question Bank
- Pre-seeded SQLite database (`interview_system.db`) mapping **20 top tech companies** and **10 specialized roles** (SDE-1, SDE-2, ML Engineer, SRE, DevOps, Systems Architect, etc.).
- Curated question tracks benchmarked against industry standards:
  - **DSA Track (Cracking the Coding Interview - CTCI)**: Conceptual reasoning, pseudocode algorithms, and tree/graph diagrams.
  - **Operating Systems Track (Silberschatz Dinosaur Book)**: Core concurrency, memory management, scheduling algorithms, and architecture diagrams.

### 2. 🧠 Candidate Knowledge Graph Synthesis
- **Resume Extraction**: Instant parsing of uploaded candidate PDF resumes via `pypdf`.
- **Public Profile Scraper (API-Free)**: Scrapes visible metrics from candidate URLs:
  - **GitHub**: Repository counts and top programming languages.
  - **LeetCode**: Problems solved (Easy / Medium / Hard) and ranking metrics.
  - **Kaggle & Portfolio Sites**: Project highlights and technology stacks.
- Builds an aggregated profile context that tailors question depth to candidate experience.

### 3. 🎨 Multimodal Answer Modes
Candidates can answer each question using their preferred format:
- **🎙️ Voice Answer**: Record audio directly in-browser (`streamlit_mic_recorder`); transcribed locally with Whisper AI, with full manual editing capability before submission.
- **⌨️ Typed Text**: Standard text input with real-time word counting.
- **💻 Pseudocode / Code**: Syntax-highlighted code editor for algorithms and logic.
- **📐 Interactive Whiteboard / Diagramming**: Built-in HTML5 canvas (`streamlit_drawable_canvas`) supporting freehand drawing, geometric shapes, lines, and color customization.
- **👁️ Vision AI Whiteboard Evaluation**: Integrates Moondream2 to analyze candidate diagrams, identifying nodes, data structures, arrows, state transitions, and memory layouts.
- **🔊 Text-to-Speech**: Candidates can click "Read question aloud" to hear questions spoken naturally.

### 4. 🛡️ Hardware Verification & Proctoring Integrity
- **Mandatory Device Pre-Flight**: Hardware verification card testing camera feed, microphone input with real-time audio VU meter, and speaker playback before unlocking the session.
- **Silent Background Facial Monitoring**: OpenCV Haar Cascade classifier tracking candidate face presence in the background.
- **Tab-Switch & Focus Loss Tracking**: Logs window blur and browser tab switches, calculating a calibrated Assessment Integrity Score.
- **Picture-in-Picture (PIP) Proctor**: Optional live proctoring HUD and fullscreen lock mode.

### 5. 📊 Comprehensive Evaluation & Actionable Report Card
- Transparent weighted scoring matrix:
  - **Technical Correctness** (35%)
  - **Reasoning & Problem-Solving** (25%)
  - **Completeness & Relevance** (15%)
  - **Code / Diagram Quality** (15%)
  - **Communication Clarity** (10%)
- **Hiring Bands**: *Strong Hire* (85+), *Hire* (70-84), *Lean Hire* (50-69), *Lean No Hire* (35-49), and *No Hire* (<35).
- Per-question 0–10 marks, specific actionable improvement tips, and ideal answer outlines.
- Diagnostic breakdown of candidate strengths, critical knowledge gaps, recommended study topics, and downloadable response sheet.

---

## 🏗 Architecture & Tech Stack

```
                     ┌───────────────────────────────────────────┐
                     │          Streamlit Web Interface          │
                     │  (Camera, Mic, Canvas, Code, Dashboard)   │
                     └─────────────────────┬─────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴─────┬──────────────────┐
         ▼                  ▼                    ▼                  ▼
┌─────────────────┐ ┌───────────────┐  ┌──────────────────┐ ┌───────────────┐
│ Speech-to-Text  │ │  Vision SLM   │  │   Text/Eval SLM  │ │ SQLite Storage│
│  Faster-Whisper │ │   Moondream2  │  │      Phi-3       │ │ question_bank │
│ (base int8 CPU) │ │ (via Ollama)  │  │  (via Ollama)    │ │ company_roles │
└─────────────────┘ └───────────────┘  └──────────────────┘ └───────────────┘
         │                  │                    │
         └──────────────────┼────────────────────┘
                            ▼
              ┌───────────────────────────┐
              │  Multi-Dimensional Report │
              │   & Integrity Scoring     │
              └───────────────────────────┘
```

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend Framework** | Streamlit | Responsive interactive web UI, state management, tabs |
| **Core SLM** | Microsoft Phi-3 (`ollama`) | Skill extraction, candidate assessment, rubric scoring |
| **Vision SLM** | Moondream2 (`ollama`) | Canvas diagram understanding and structural analysis |
| **Speech-to-Text** | `faster-whisper` | Low-latency local voice transcription |
| **Text-to-Speech** | `pyttsx3` | Offline natural question narration |
| **Audio Processing** | `pydub`, `FFmpeg` | Audio decoding and stream conversions |
| **Whiteboard Canvas** | `streamlit-drawable-canvas` | Vector drawing and diagram input |
| **Proctoring Engine** | `OpenCV`, `streamlit-webrtc` | Haar Cascade face detection and camera monitoring |
| **Document Parsing** | `pypdf`, `BeautifulSoup4` | PDF resume parsing and profile web scraping |
| **Database** | SQLite3 (`interview_system.db`) | Role taxonomies and curated question banks |

---

## 💻 System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python**: Version 3.10 to 3.12.
- **RAM**: Minimum 8 GB (16 GB recommended for concurrent SLM inference).
- **Disk Space**: ~6 GB for Ollama models (`phi3` and `moondream`) + Whisper weights.
- **Hardware**: Working webcam and microphone.
- **Dependencies**: FFmpeg installed on system PATH.

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/SiriKadavala1925/AI_Interview_System.git
cd AI_Interview_System
```

### 2. Install Local SLMs via Ollama
1. Download and install [Ollama](https://ollama.ai/).
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. Pull the required models:
   ```bash
   ollama pull phi3
   ollama pull moondream
   ```

### 3. Install FFmpeg
The audio transcription pipeline requires `ffmpeg` and `ffprobe` for audio conversions.

- **Windows**:
  - Download a build from [gyan.dev FFmpeg](https://www.gyan.dev/ffmpeg/builds/) or install via Scoop/Chocolatey:
    ```powershell
    winget install Gyan.FFmpeg
    # or
    choco install ffmpeg
    ```
  - *Note*: If placed in `C:\FFmpeg\bin\ffmpeg.exe`, the application automatically resolves it.
- **macOS**:
  ```bash
  brew install ffmpeg
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```

### 4. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install pyttsx3 numpy requests beautifulsoup4 pydub faster-whisper streamlit-mic-recorder streamlit-drawable-canvas pillow
```

*(Optional for proctoring camera & face detection)*:
```bash
pip install streamlit-webrtc opencv-python av
```

### 5. Initialize the SQLite Database
The SQLite database stores company roles and questions. Initialize or reset it by running:
```bash
cd ai_interview_system
python database.py
```
*Output will indicate database seeding with over 4,000 question combinations across DSA and OS tracks.*

---

## 🏃 Running the Application

### Option A: Using the One-Click PowerShell Launcher (Windows)
Double-click `run_app.ps1` inside the `ai_interview_system/` directory or run in PowerShell:
```powershell
.\ai_interview_system\run_app.ps1
```

### Option B: Manual Command Line
```bash
cd ai_interview_system
streamlit run app.py
```

Once started, open your browser and navigate to:
```
http://localhost:8501
```

---

## 🔄 Interview Workflow

1. **Candidate Setup & Knowledge Graph**:
   - Select target company (e.g., Google, Microsoft, Amazon) and job role.
   - Upload candidate resume in PDF format.
   - *(Optional)* Add public profile links (GitHub, LeetCode, Kaggle, Portfolio). The system builds a personalized candidate profile.
2. **Proctor & Hardware Verification**:
   - Complete the mandatory microphone test (live audio VU meter), camera check, and audio playback test.
3. **Interactive Assessment**:
   - The system presents questions sequentially.
   - Listen to the question via text-to-speech.
   - Choose answer mode: **Voice** (Whisper STT), **Text**, **Code**, or **Diagram** (Canvas + Vision AI).
   - Instant vision feedback is available for diagrams.
4. **Final Scoring & Feedback**:
   - Microsoft Phi-3 analyzes the entire session transcript against the standard rubric.
   - View hiring verdict, radar component scores, question-by-question tips, integrity metrics, and download the full report.

---

## 📐 Evaluation & Scoring Rubric

The scoring engine evaluates candidate performance using an objective, uninflated grading standard:

```
Total Score (100%) =
    (Technical Correctness   × 0.35) +
    (Reasoning & Decomposition × 0.25) +
    (Completeness & Relevance × 0.15) +
    (Code / Diagram Quality   × 0.15) +
    (Communication Clarity    × 0.10)
```

### Hiring Bands
| Band | Score Range | Description |
| :--- | :---: | :--- |
| 🟢 **Strong Hire** | 85 – 100 | Exceptional technical depth, structured communication, and solid problem decomposition. |
| 🔵 **Hire** | 70 – 84 | Solid fundamentals across topics; ready for production contributions with standard onboarding. |
| 🟡 **Lean Hire** | 50 – 69 | Borderline; demonstrates competency in core areas but shows notable conceptual or implementation gaps. |
| 🟠 **Lean No Hire** | 35 – 49 | Below the expected bar for the role; incomplete answers or significant conceptual misunderstandings. |
| 🔴 **No Hire** | 0 – 34 | Candidate did not demonstrate readiness; critical concepts require substantial remediation. |

---

## 📁 Directory Structure

```
AI_INTERVIEW_SYSTEM/
│
├── .gitignore                      # Git ignore file (virtual envs, DBs, cache)
├── requirements.txt                # Core Python package requirements
├── README.md                       # Complete system documentation
│
└── ai_interview_system/            # Main application package
    ├── app.py                      # Primary Streamlit application & evaluation workflow
    ├── database.py                 # SQLite database schema, seeding, and question bank
    ├── run_app.ps1                 # Windows PowerShell one-click launcher
    ├── interview_system.db         # Pre-seeded SQLite database file
    │
    ├── benchmark_eval.py           # Evaluation pipeline benchmark script for Phi-3
    ├── check_subjects.py           # Utility to inspect question categories
    ├── test_canvas.py              # Unit test for interactive canvas drawing
    ├── test_ollama.py              # Connectivity check for local Ollama service
    └── test_vision.py              # Visual model integration test for Moondream2
```

---

## 🧪 Testing & Verification Utilities

The repository includes standalone validation scripts to verify individual subsystems before running the full interview flow:

- **Check Ollama Connection**:
  ```bash
  python ai_interview_system/test_ollama.py
  ```
- **Test Vision Model (Moondream2)**:
  ```bash
  streamlit run ai_interview_system/test_vision.py
  ```
- **Test Drawing Canvas**:
  ```bash
  streamlit run ai_interview_system/test_canvas.py
  ```
- **Benchmark Evaluation Speed**:
  ```bash
  python ai_interview_system/benchmark_eval.py
  ```

---

## ❓ Troubleshooting & FAQs

### 1. `Ollama connection failed` or `model not found`
- Ensure the Ollama daemon is active (`ollama serve`).
- Verify installed models by running `ollama list`.
- Pull missing models using `ollama pull phi3` and `ollama pull moondream`.

### 2. `ffprobe.exe / ffmpeg.exe missing`
- Ensure FFmpeg binaries are installed and accessible on your system `PATH`.
- On Windows, verify that `ffmpeg.exe` and `ffprobe.exe` exist in `C:\FFmpeg\bin` or update the path in `app.py`.

### 3. Voice transcription is blank or returning warnings
- Check browser permissions for microphone access.
- In the Hardware Verification stage, ensure the audio VU meter moves when speaking.

### 4. Camera or face detection unavailable
- Install the optional WebRTC and computer vision libraries:
  ```bash
  pip install streamlit-webrtc opencv-python av
  ```
- If OpenCV is installed without Haar cascade files, the application automatically handles fallback gracefully without breaking the interview session.

---

## 📄 License

This project is created for educational and technical interview preparation purposes.
