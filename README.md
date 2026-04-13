# 🎙️ VoiceAgent — Voice-Controlled Local AI Agent

A voice-controlled AI agent that accepts audio input, transcribes speech using **OpenAI Whisper** (local), classifies intent via **Ollama** (local LLM with Groq API fallback), executes local tools, and displays results in a polished **Streamlit** chat UI — all running entirely on your own machine.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-FF4B4B?logo=streamlit&logoColor=white)
![Whisper](https://img.shields.io/badge/Whisper-Local_STT-green?logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-blue)

---

## 📋 Table of Contents

- [Architecture](#️-architecture)
- [Features](#-features)
- [Setup Instructions](#-setup-instructions)
- [Running the App](#-running-the-app)
- [Usage Guide](#-usage-guide)
- [UI Pipeline](#-ui-pipeline-stages)
- [Supported Intents](#-supported-intents)
- [Project Structure](#-project-structure)
- [Hardware Workarounds](#️-hardware-workarounds)
- [Tech Stack](#️-tech-stack)

---

## 🏗️ Architecture

```
┌──────────────────┐
│  Audio Input     │  ← Microphone recording or uploaded audio file
│  (Mic / Upload)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Whisper STT     │  ← OpenAI Whisper running locally on GPU (CUDA)
│  (Local, GPU)    │    Supports: base / small / medium models
└────────┬─────────┘
         │ Transcribed Text
         ▼
┌──────────────────┐
│  Preview & Edit  │  ← User reviews/corrects the transcription before processing
│  (Human-in-loop) │    Can edit text or click Re-record
└────────┬─────────┘
         │ Confirmed Text
         ▼
┌──────────────────────────────────┐
│  Intent Classifier               │  ← Ollama (Gemma 4, local) — primary
│  Ollama (local) / Groq (fallback)│     Groq API (Llama 3.3 70B) — fallback
└────────┬─────────────────────────┘
         │ JSON: { intent, parameters }
         ▼
┌──────────────────┐
│  Tool Executor   │  ← Dispatch to the right tool based on intent
│  (Sandboxed)     │    All file ops restricted to output/ directory
└────────┬─────────┘
         │ ExecutionResult
         ▼
┌──────────────────────────────────────────┐
│  Streamlit Chat UI                       │
│  ┌─────────────────────────────────────┐ │
│  │ Transcribed Text: "..."             │ │
│  │ Detected Intent:  💬 General Chat   │ │
│  │ Action Taken:     Generated response│ │
│  │ FINAL OUTPUT                        │ │
│  │ [Response text / code / summary]    │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Description |
|-------|-------------|
| `idle` | Waiting for input |
| `preview` | Audio transcribed; user reviews/edits text |
| `classifying` | LLM runs intent classification with a spinner |
| `confirm` | High-risk actions (file creation) shown for user approval |
| `execute` | Tool runs; results streamed live to the chat UI |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 Dual Audio Input | Record via microphone or upload audio files (.wav, .mp3, .m4a, .ogg, .flac, .webm) |
| 🔍 Transcription Preview | Shows transcribed text for review/editing BEFORE the AI acts on it |
| 🔊 Local Whisper STT | GPU-accelerated speech-to-text with configurable model size |
| 🧠 Local LLM (Ollama) | Intent classification + code generation via Gemma 4 |
| ⚡ Groq API Fallback | Automatic fallback to Groq (Llama 3.3 70B) when Ollama is offline |
| 📁 File Operations | Create files and folders, sandboxed to `output/` |
| 💻 Code Generation | Generate code in any language and save to file |
| 📝 Text Summarization | Summarize provided content via LLM |
| 💬 General Chat | Conversational AI responses, always in English |
| 🛡️ Path Sandboxing | All file operations restricted to `output/` with traversal attack prevention |
| 🔗 Compound Commands | Handles multi-intent requests (e.g., "Write code and save it") |
| 📜 Session History | Full action history in sidebar with timestamps |
| ✅ Human-in-the-Loop | Confirmation dialog before any file system modification |

---

## 🚀 Setup Instructions

### Prerequisites

Before starting, ensure you have the following installed:

- ✅ **Python 3.11+** — [python.org](https://www.python.org/downloads/)
- ✅ **Ollama** — [ollama.ai](https://ollama.ai) (for local LLM)
- ✅ **FFmpeg** — Required by Whisper for audio decoding
- ✅ **NVIDIA GPU with CUDA** *(recommended)* — For fast Whisper transcription

#### Installing FFmpeg (Windows)

```bash
# Option 1: Using winget
winget install ffmpeg

# Option 2: Using Chocolatey
choco install ffmpeg

# Option 3: Manual install
# Download from https://ffmpeg.org/download.html → add to PATH
```

Verify it works:
```bash
ffmpeg -version
```

---

### Step 1 — Clone the Repository

```bash
git clone <your-repo-url>
cd VoiceAgent
```

---

### Step 2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `streamlit>=1.33.0` — UI framework
- `openai-whisper` — Local speech-to-text
- `torch` — PyTorch (used by Whisper for GPU inference)
- `groq` — Groq API client (LLM fallback)
- `requests` — Ollama API communication
- `python-dotenv` — Environment variable loading

> **Note (Windows + CUDA):** If `torch` is installed without CUDA support, Whisper will fall back to CPU which is significantly slower. To install PyTorch with CUDA:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Step 3 — Setup Ollama (Local LLM)

Install Ollama from [ollama.ai](https://ollama.ai), then pull the required model:

```bash
ollama pull gemma4
```

Start the Ollama server (it usually starts automatically):
```bash
ollama serve
```

Verify it's running:
```bash
ollama list
# Should show: gemma4:latest
```

The app checks if Ollama is available at startup and shows its status in the sidebar.

---

### Step 4 — Configure Environment (Optional — Groq Fallback)

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_key_here
```

You can also enter the Groq API key directly in the app sidebar at any time. A Groq API key is **not required** if Ollama is running.

Get a free Groq API key at [console.groq.com](https://console.groq.com).

---

## ▶️ Running the App

```bash
python -m streamlit run app.py
```

> **Important (Windows):** Use `python -m streamlit run app.py` instead of `streamlit run app.py`. The bare `streamlit` command may not be found if Streamlit is not added to your `PATH`.

The app opens at **http://localhost:8501**.

---

## 🎯 Usage Guide

### Recording a Voice Command

1. Click the **microphone button** in the audio recorder and speak your command
2. Click **➤ Send Voice Command**
3. The app transcribes your speech and shows a **🔍 Review Transcription** panel
4. Read the transcribed text — **edit it** if Whisper misheard anything
5. Click **✅ Confirm & Process** to send it to the AI
6. The result appears in the chat UI with all 4 fields populated

### Uploading an Audio File

1. Click **▼ Upload an audio file instead** to expand the uploader
2. Upload a `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, or `.webm` file
3. Click **🚀 Process Uploaded File**
4. Same preview → confirm → execute flow applies

### Typing a Command

Use the **"Or type your command here..."** chat input at the bottom of the page. Text commands skip the preview step (you can already read what you typed) and go straight to classification.

### What the UI Always Shows

Every response in the chat displays exactly these four fields:

```
Transcribed Text:  "your original command"
Detected Intent:   💬 General Chat  (or 📁 / 💻 / 📝)
Action Taken:      Generated conversational response
FINAL OUTPUT
[The actual result — text / code / file confirmation]
```

### Example Commands

| Voice Command | Detected Intent | Output |
|--------------|----------------|--------|
| "Create a file called notes.txt" | 📁 Create File | `output/notes.txt` created |
| "Write a Python function to sort a list" | 💻 Write Code | Code generated → `output/sort_a_list.py` |
| "Summarize the theory of relativity" | 📝 Summarize | Concise summary shown |
| "What is machine learning?" | 💬 General Chat | Conversational answer |
| "Create a Python file with a retry decorator" | 💻 Write Code | Code → `output/retry_decorator.py` |
| "Write a sorting function and save it" | 💻 Write Code + 📁 Create File | Compound command: generates then saves |

---

## 🎯 Supported Intents

| Intent | Trigger Examples | Action Taken |
|--------|-----------------|--------------|
| `create_file` | "Create a file named X", "Make a folder called Y" | Creates empty file or directory in `output/` |
| `write_code` | "Write a function to...", "Generate Python code for..." | Generates code via LLM → saves to `output/` |
| `summarize` | "Summarize this...", "Give me a summary of..." | Summarizes content via LLM |
| `general_chat` | "What is...", "Hello", "Explain...", "Tell me about..." | Conversational AI response |

> Compound commands (e.g., "Write a function and save it to a file") are automatically split into multiple sequential actions.

---

## 📁 Project Structure

```
VoiceAgent/
├── app.py                  # Main Streamlit UI — pipeline stages, rendering, input handling
├── stt_engine.py           # Whisper speech-to-text engine (local, GPU-accelerated)
├── intent_classifier.py    # Ollama + Groq intent classification and LLM calling
├── tool_executor.py        # Tool dispatch: file creation, code gen, summarize, chat
├── config.py               # All configuration: models, prompts, intent enum, paths
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Your local secrets (not committed to git)
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── output/                 # Sandboxed directory for all generated files
    └── .gitkeep
```

### Module Responsibilities

**`app.py`** — Orchestrates the full pipeline via Streamlit session state stages (`idle → preview → classifying → confirm → execute`). Renders the 4-field chat UI for every interaction.

**`stt_engine.py`** — Loads OpenAI Whisper locally, transcribes audio bytes to text. Handles GPU/CPU fallback automatically.

**`intent_classifier.py`** — Calls the LLM (Ollama or Groq) with a structured system prompt and parses the JSON response to extract intents and parameters.

**`tool_executor.py`** — Dispatches to the correct handler (`execute_create_file`, `execute_write_code`, `execute_summarize`, `execute_general_chat`). Returns a typed `ExecutionResult` dataclass.

**`config.py`** — Central configuration: model names, Ollama URL, system prompts, `Intent` enum, and the `output/` directory path.

---

## 🖥️ Hardware Workarounds

### 1. Whisper — GPU vs CPU Fallback

**Setup:** OpenAI Whisper was designed to run on NVIDIA GPUs via CUDA. On machines without a CUDA-capable GPU, Whisper silently falls back to CPU inference, which is 5–10× slower.

**Workaround implemented in `stt_engine.py`:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model(model_name, device=device)
```

- **With GPU (CUDA):** `base` model transcribes a 5-second clip in ~0.5s
- **Without GPU (CPU):** Use the `base` model (smallest). The `small`/`medium` models will be very slow on CPU
- **Recommendation:** Select `base` in the sidebar if running on CPU only

**Tested on:**
| Hardware | Whisper Model | Transcription Time |
|----------|--------------|-------------------|
| RTX 3050 6GB (CUDA 12.1) | `small` | ~1.5s |
| CPU only | `base` | ~6-10s |

---

### 2. LLM — Ollama Unavailable Workaround (Groq Fallback)

**Problem:** Ollama requires a running local server and enough RAM/VRAM to load the model (Gemma 4 requires ~8GB RAM). On resource-constrained machines, Ollama may not be available.

**Workaround:** A dual-backend system with automatic fallback:

```
User Preference: "Auto" (default)
       │
       ▼
Is Ollama running? ──Yes──▶ Use Ollama (Gemma 4) [local, private, no limits]
       │
      No
       │
       ▼
Is Groq API key set? ──Yes──▶ Use Groq API (Llama 3.3 70B) [cloud, free tier]
       │
      No
       │
       ▼
Return error to user with clear message
```

Users can also manually override in the sidebar:
- **Auto** — Prefer Ollama, fallback to Groq
- **Ollama Only** — Fail if Ollama is not available
- **Groq API Only** — Always use Groq (requires API key)

---

### 3. Material Icons Font — CSS Fallback Glitch

**Problem:** Streamlit's `st.expander` widget uses Google's Material Icons CDN font for its expand/collapse arrow. When the font fails to load (offline environments, CDN blocked, slow connections), the raw ligature text `_arrow_right_` renders visibly on top of the label text.

**Workaround:** Replaced `st.expander` entirely with a plain `st.button` toggle that uses Unicode characters (`▼` / `▲`) to indicate expand/collapse — no external font dependency whatsoever.

---

### 4. Whisper Language Detection — Non-English Audio

**Problem:** When audio contains non-English speech (e.g., Urdu, Hindi), Whisper may transcribe it in that language's script. The LLM, seeing non-English text in the conversation context, may then respond in the same language even if the user's intent was English.

**Workaround 1 — Transcription Preview:** The new **🔍 Review Transcription** step lets users catch and correct any misrecognized text before it is sent to the LLM.

**Workaround 2 — English-enforced system prompt:**
```python
CHAT_PROMPT = """...
CRITICAL RULE: Always respond in English unless the user explicitly asks you to speak another language.
Do NOT write in Urdu or any other language just because the previous conversation history contained it."""
```

---

### 5. `streamlit` Command Not Found on Windows

**Problem:** After `pip install streamlit`, the `streamlit` command may not be available in PowerShell if Python's `Scripts/` folder is not on the system `PATH`.

**Workaround:** Always run using the Python module flag:
```bash
python -m streamlit run app.py
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| UI Framework | Streamlit | 1.56+ |
| Speech-to-Text | OpenAI Whisper (local) | Latest |
| LLM Primary | Ollama + Gemma 4 | Latest |
| LLM Fallback | Groq API + Llama 3.3 70B | Latest |
| GPU Acceleration | PyTorch + CUDA | 2.x + CUDA 12.1 |
| Language | Python | 3.11+ |
| Audio Processing | FFmpeg (via Whisper) | Latest |

---

## 🌟 Bonus Features

| Feature | Status | Description |
|---------|--------|-------------|
| Transcription Preview | ✅ | User reviews/edits speech-to-text output before processing |
| Compound Commands | ✅ | Multi-intent requests parsed into sequential action arrays |
| Human-in-the-Loop | ✅ | Confirmation prompt before any file system modification |
| Graceful Degradation | ✅ | Handles bad audio, API failures, and JSON parse errors cleanly |
| Session Memory | ✅ | Last 3 interactions used as context for follow-up commands |
| LLM Fallback | ✅ | Automatic Ollama → Groq fallback with visual status indicator |
| Path Sandboxing | ✅ | All file ops restricted to `output/` with traversal attack prevention |
| English Enforcement | ✅ | LLM always responds in English regardless of input language |

---

## 📜 License

MIT License — free to use, modify, and distribute.
