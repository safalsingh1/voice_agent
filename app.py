"""
app.py — Voice-Controlled Local AI Agent — Streamlit UI.

A polished interface for audio input, speech-to-text, intent classification,
and tool execution with session history.
"""

import os
import streamlit as st
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── Security constants (configurable via .env) ────────────────────────────────
# Maximum allowed audio file size in MB
MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "25"))
MAX_AUDIO_SIZE_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024

# Rate limiting: maximum requests per window
RATE_LIMIT_MAX: int = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

from config import (
    WHISPER_MODELS,
    DEFAULT_WHISPER_MODEL,
    INTENT_LABELS,
    Intent,
    SUPPORTED_AUDIO_FORMATS,
)
from stt_engine import transcribe_audio
from intent_classifier import classify_intent, _check_ollama_available
from tool_executor import execute_tool

# Load .env file for auto-populating API keys
load_dotenv()

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceAgent — Voice-Controlled AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

/* Fix Streamlit Material Icons fallback: hide broken raw icon text on expander arrows */
[data-testid="stExpander"] summary svg { display: inline-block !important; }
[data-testid="stExpander"] summary span[class*="icon"] {
    font-family: 'Material Icons', sans-serif !important;
    font-size: 1.2rem !important;
    vertical-align: middle;
    overflow: hidden;
    max-width: 1.5rem;
}
/* Force expander summary to not wrap/overflow broken icon text */
[data-testid="stExpander"] > details > summary {
    display: flex;
    align-items: center;
    overflow: hidden;
}
[data-testid="stExpander"] > details > summary p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.sidebar-title { font-size: 0.85rem; font-weight: 600; color: rgba(255, 255, 255, 0.5); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 0.75rem; }
.history-item { padding: 0.75rem 1rem; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 10px; margin-bottom: 0.5rem; font-size: 0.85rem; transition: background 0.2s; }
.history-item:hover { background: rgba(255, 255, 255, 0.06); }
.history-time { color: rgba(255, 255, 255, 0.4); font-size: 0.75rem; }
.history-intent { font-weight: 600; margin: 0.2rem 0; }
.history-text { color: rgba(255, 255, 255, 0.6); font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.result-card { background: rgba(255,255,255,0.03); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-top: 0.5rem; }
.result-header { font-size: 0.9rem; font-weight: 600; color: #a5b4fc; text-transform: uppercase; margin-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.3rem; }
.chat-field { margin-bottom: 0.8rem; }
.chat-field strong { color: rgba(255,255,255,0.8); display: inline-block; width: 140px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "history": [],
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "llm_preference": "Auto",
        "pipeline_stage": "idle",
        "pipeline_stt": None,
        "pipeline_intent": None,
        "show_uploader": False,
        # Rate-limiting: track request timestamps within the sliding window
        "rate_limit_timestamps": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── Rate Limiter ─────────────────────────────────────────────────────────────
def _check_rate_limit() -> bool:
    """
    Returns True if the request is allowed, False if rate limit is exceeded.
    Uses a sliding-window approach per browser session.
    """
    now = datetime.now()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW_SECONDS)

    # Prune timestamps outside the window
    st.session_state.rate_limit_timestamps = [
        ts for ts in st.session_state.rate_limit_timestamps
        if ts > window_start
    ]

    if len(st.session_state.rate_limit_timestamps) >= RATE_LIMIT_MAX:
        return False

    st.session_state.rate_limit_timestamps.append(now)
    return True


# ─── Audio Size Guard ─────────────────────────────────────────────────────────
def _check_audio_size(audio_bytes: bytes) -> bool:
    """Return True if audio is within the allowed size limit."""
    return len(audio_bytes) <= MAX_AUDIO_SIZE_BYTES

init_session_state()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with configuration and history."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🎙️</span>
            <h2 style="margin: 0.3rem 0 0; font-weight: 700; font-size: 1.3rem; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">VoiceAgent</h2>
        </div><hr/>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">⚡ LLM Backend Status</div>', unsafe_allow_html=True)
        if _check_ollama_available():
            st.markdown('<span class="status-dot" style="background:#4CAF50;"></span> <span style="color:#81C784; font-weight:500;">Ollama Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot" style="background:#F44336;"></span> <span style="color:#EF9A9A; font-weight:500;">Ollama Offline</span>', unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🔑 Groq API Key (Fallback)</div>', unsafe_allow_html=True)
        groq_key = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password", label_visibility="collapsed", placeholder="gsk_...")
        st.session_state.groq_api_key = groq_key

        st.markdown("---")
        st.markdown('<div class="sidebar-title">⚙️ LLM Backend Preference</div>', unsafe_allow_html=True)
        backends = ["Auto", "Ollama Only", "Groq API Only"]
        preference = st.radio("backend", backends, index=backends.index(st.session_state.llm_preference), label_visibility="collapsed")
        st.session_state.llm_preference = preference
        st.caption("Auto: Prefer local Ollama, fallback to Groq.")

        st.markdown("---")
        st.markdown('<div class="sidebar-title">🔊 Whisper STT Model</div>', unsafe_allow_html=True)
        model = st.selectbox("Whisper Model", WHISPER_MODELS, index=WHISPER_MODELS.index(st.session_state.whisper_model), label_visibility="collapsed")
        st.session_state.whisper_model = model
        st.caption(f"Using GPU-accelerated `{model}` model")

        st.markdown("---")
        st.markdown('<div class="sidebar-title">📜 Session History</div>', unsafe_allow_html=True)
        if st.session_state.history:
            for entry in reversed(st.session_state.history):
                icon, label, color = INTENT_LABELS.get(entry["intent"], ("❓", "Unknown", "#666"))
                st.markdown(f'<div class="history-item"><div class="history-time">{entry["timestamp"]}</div><div class="history-intent" style="color:{color};">{icon} {label}</div><div class="history-text">{entry["transcript"][:60]}...</div></div>', unsafe_allow_html=True)
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No actions yet. Start by recording or uploading audio!")

# ─── Chat Processing Logic ──────────────────────────────────────────────────────
def process_input(audio_bytes: bytes = None, text_input: str = None):
    """Process either audio or text input and kick off the STT/Intent pipeline."""
    transcript = ""

    # ── Security: rate limit check ────────────────────────────────────────────
    if not _check_rate_limit():
        st.error(
            f"⏱️ Rate limit reached: maximum {RATE_LIMIT_MAX} requests per "
            f"{RATE_LIMIT_WINDOW_SECONDS} seconds. Please wait a moment."
        )
        return

    if audio_bytes:
        # ── Security: audio size check ────────────────────────────────────────
        if not _check_audio_size(audio_bytes):
            size_mb = len(audio_bytes) / (1024 * 1024)
            st.error(
                f"🚫 Audio file is too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_AUDIO_SIZE_MB} MB."
            )
            return
        with st.spinner("🔊 Transcribing audio..."):
            stt_result = transcribe_audio(audio_bytes, st.session_state.whisper_model)
        
        if not stt_result["success"] or not stt_result["text"].strip():
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "transcript": "(Audio recording)",
                "intent": Intent.GENERAL_CHAT,
                "action": "Transcription Failed",
                "success": False,
                "error": stt_result.get("error", "No speech detected.")
            })
            st.session_state.pipeline_stage = "idle"
            return
            
        transcript = stt_result["text"]
        # Pause at 'preview' so the user can read/edit before we classify
        st.session_state.pending_transcript = transcript
        st.session_state.pipeline_stage = "preview"
        st.rerun()

    elif text_input:
        # Text input doesn't need preview — go straight to classifying
        st.session_state.pending_transcript = text_input
        st.session_state.pipeline_stage = "classifying"
        st.rerun()


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    """Main application entry point."""
    render_sidebar()

    st.markdown("""<div style="text-align:center; padding: 1rem 0 2rem;">
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin:0;">Word 2 Code</h1>
    </div>""", unsafe_allow_html=True)

    # 1. Render Chat History
    if not st.session_state.history and st.session_state.pipeline_stage == "idle":
        st.markdown("""<div style="text-align:center; padding:3rem 1rem; color:rgba(255,255,255,0.3);"><div style="font-size:4rem; margin-bottom:1rem;">🎤</div><h3 style="margin:0; font-weight:600; color:rgba(255,255,255,0.4);">Ready</h3><p style="margin:0.5rem 0 0; font-size:0.95rem;">Speak or type below to begin.</p></div>""", unsafe_allow_html=True)

    for entry in st.session_state.history:
        # User input is shown contextually within the assistant's structured response below
        with st.chat_message("assistant", avatar="🤖"):
            icon, label, _ = INTENT_LABELS.get(entry.get("intent", Intent.GENERAL_CHAT), ("❓", "Unknown", "#666"))
            
            # Explicitly formatted UI as requested
            st.markdown(f"**Transcribed Text:** `{entry.get('transcript', '(Audio recording)')}`")
            st.markdown(f"**Detected Intent:** {icon} {label}")
            st.markdown(f"**Action Taken:** {entry.get('action', 'Unknown')}")
            st.caption("FINAL OUTPUT")
            if not entry.get("success"):
                st.error(f"Task Failed: {entry.get('action', 'Unknown')}\n\nError details: {entry.get('error', 'Unknown')}")
            else:
                if entry.get("file_path"):
                     st.caption(f"📂 Saved to: `{entry['file_path']}`")
                     
                if entry.get("full_output"):
                    if entry.get("intent") == Intent.WRITE_CODE:
                        st.code(entry["full_output"])
                    elif entry.get("intent") == Intent.SUMMARIZE:
                        st.markdown(f"> {entry['full_output']}")
                    else:
                        st.markdown(entry["full_output"])
                else:
                    st.info("Task completed successfully. No text output generated.")
                
            st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;' />", unsafe_allow_html=True)

    # 2a. Preview stage: show transcription in an editable box before classifying
    if st.session_state.pipeline_stage == "preview":
        st.markdown("### 🔍 Review Transcription")
        st.info("⚠️ Please review what was heard. Edit if needed, then click **Confirm & Process**.")
        edited = st.text_area(
            "Transcribed Text (edit if needed):",
            value=st.session_state.get("pending_transcript", ""),
            height=80,
            key="preview_text_area"
        )
        col1, col2 = st.columns(2)
        if col1.button("✅ Confirm & Process", type="primary", use_container_width=True):
            st.session_state.pending_transcript = edited.strip()
            st.session_state.pipeline_stage = "classifying"
            st.rerun()
        if col2.button("🔁 Re-record", use_container_width=True):
            st.session_state.pipeline_stage = "idle"
            st.session_state.audio_key = st.session_state.get("audio_key", 0) + 1
            st.rerun()

    # 2b. Classifying stage: show transcript immediately, then classify intent
    if st.session_state.pipeline_stage == "classifying":
        transcript = st.session_state.get("pending_transcript", "")
        groq_key = st.session_state.groq_api_key or None

        # Show the transcribed text NOW so user can read it while we think
        with st.chat_message("user"):
            st.markdown(transcript)

        # Build context from history
        recent = st.session_state.history[-3:]
        context_str = ""
        for entry in recent:
            if entry.get("success"):
                out = entry.get("full_output", "")[:1000]
                context_str += f"\nPrevious Request: {entry['transcript']}\nPrevious Output: {out}\n"
        transcript_context = f"=== CONVERSATION HISTORY ==={context_str}\n=== CURRENT NEW REQUEST ===\n{transcript}" if context_str else transcript
        st.session_state.pending_transcript_context = transcript_context

        # Classify intent with a visible spinner
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Thinking..."):
                intent_result = classify_intent(transcript_context, groq_key, st.session_state.llm_preference)

        actions = intent_result.get("actions", [])
        needs_confirmation = any(a["intent"] in (Intent.CREATE_FILE, Intent.WRITE_CODE) for a in actions)
        st.session_state.pending_actions = actions

        if needs_confirmation:
            st.session_state.pipeline_stage = "confirm"
        else:
            st.session_state.pipeline_stage = "execute"
        st.rerun()

    if st.session_state.pipeline_stage == "confirm":
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("⚠️ This command will modify your file system. Proceed?")
            
            transcript = st.session_state.pending_transcript
            actions = st.session_state.pending_actions
            if actions:
                intent = actions[0]["intent"]
                icon, label, _ = INTENT_LABELS.get(intent, ("❓", "Unknown", "#666"))
                action_preview = f"Pending execution: {label}"
            else:
                icon, label = "❓", "Unknown"
                action_preview = "Unknown pending action"
                
            st.markdown(f"**Transcribed Text:** `{transcript}`")
            st.markdown(f"**Detected Intent:** {icon} {label}")
            st.markdown(f"**Action Taken:** {action_preview}")
            st.caption("FINAL OUTPUT")
            st.info("Awaiting your confirmation to proceed...")

            col1, col2 = st.columns(2)
            if col1.button("✅ Proceed", type="primary"):
                st.session_state.pipeline_stage = "execute"
                st.rerun()
            if col2.button("❌ Cancel"):
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "transcript": st.session_state.pending_transcript,
                    "intent": Intent.GENERAL_CHAT,
                    "action": "Action Cancelled by User",
                    "success": False,
                    "error": None
                })
                st.session_state.pipeline_stage = "idle"
                st.session_state.pending_actions = []
                st.rerun()

    # 3. Execution Pipeline Loop inside Assistant Message
    if st.session_state.pipeline_stage == "execute":
        with st.chat_message("user"):
            st.markdown(st.session_state.get("pending_transcript", ""))
        with st.chat_message("assistant", avatar="🤖"):
            groq_key = st.session_state.groq_api_key or None
            last_output = ""
            actions = st.session_state.pending_actions
            transcript = st.session_state.pending_transcript
            
            for idx, act in enumerate(actions):
                intent = act["intent"]
                params = act["parameters"]
                icon, label, color = INTENT_LABELS.get(intent, ("❓", "Unknown", "#666"))
                
                # Context passing for compound commands natively
                if idx > 0 and intent in (Intent.WRITE_CODE, Intent.CREATE_FILE):
                    if not params.get("description") or params.get("description") == transcript:
                        params["description"] = f"Using this content: {last_output}"
                        
                with st.spinner(f"⚙️ Executing {idx+1}/{len(actions)}: {label}..."):
                    exec_result = execute_tool(intent, params, st.session_state.pending_transcript_context, groq_key, st.session_state.llm_preference)
                    
                last_output = exec_result.output_content
                
                # Render results live as they complete
                st.markdown(f"**Transcribed Text:** `{transcript if idx == 0 else f'(Action {idx+1})'}`")
                st.markdown(f"**Detected Intent:** {icon} {label}")
                st.markdown(f"**Action Taken:** {exec_result.action_description}")
                st.caption("FINAL OUTPUT")
                if not exec_result.success:
                    st.error(f"Execution Failed: {exec_result.action_description}\n\nError details: {exec_result.error}")
                else:
                    if exec_result.file_path:
                        st.caption(f"📂 Saved to: `{exec_result.file_path}`")
                    
                    if exec_result.output_content:
                        if intent == Intent.WRITE_CODE:
                            st.code(exec_result.output_content, language=(params.get("language") or "python").lower())
                        elif intent == Intent.SUMMARIZE:
                            st.markdown(f"> {exec_result.output_content}")
                        else:
                            st.markdown(exec_result.output_content)
                    else:
                        st.info("Task completed successfully. No text output generated.")
                        
                st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 2rem 0;' />", unsafe_allow_html=True)
                    
                # Save execution history individually
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "transcript": transcript if idx == 0 else f"(Action {idx+1})",
                    "intent": intent,
                    "action": exec_result.action_description,
                    "success": exec_result.success,
                    "full_output": exec_result.output_content,
                    "file_path": exec_result.file_path,
                    "error": exec_result.error
                })
                
                if not exec_result.success:
                    break
                    
        st.session_state.pipeline_stage = "idle"
        st.rerun()

    # Tracking previous stage
    st.session_state._prev_stage = st.session_state.pipeline_stage

    # ─── Bottom Input Area ────────────────────────────────────────────────────
    st.markdown("---")
    
    # Ensure a fresh widget key each cycle to avoid stale audio state
    if "audio_key" not in st.session_state:
        st.session_state.audio_key = 0
    
    # Mic input
    audio_val = st.audio_input(
        "🎙️ Record your command", 
        label_visibility="collapsed",
        key=f"audio_input_{st.session_state.audio_key}"
    )
    
    # Send button
    send_clicked = st.button("➤ Send Voice Command", type="primary", use_container_width=True, help="Send recorded voice command")
    
    # File upload toggle (manual, avoids broken st.expander Material Icons font)
    toggle_label = "▲ Hide file uploader" if st.session_state.show_uploader else "▼ Upload an audio file instead"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader(
            "Upload audio file",
            type=[fmt.lstrip(".") for fmt in SUPPORTED_AUDIO_FORMATS],
            label_visibility="collapsed",
            key=f"file_upload_{st.session_state.audio_key}"
        )
        if uploaded_file:
            st.audio(uploaded_file.getvalue())
            if st.button("🚀 Process Uploaded File", type="primary", use_container_width=True):
                st.session_state.show_uploader = False
                process_input(audio_bytes=uploaded_file.getvalue())
                st.session_state.audio_key += 1
                st.rerun()
    
    # ─── Process Inputs ───────────────────────────────────────────────────────
    text_val = st.chat_input("Or type your command here...")
    
    if st.session_state.pipeline_stage == "idle":
        if send_clicked and audio_val:
            process_input(audio_bytes=audio_val.getvalue())
            st.session_state.audio_key += 1
            st.rerun()
        elif text_val:
            process_input(text_input=text_val)
            st.rerun()

if __name__ == "__main__":
    main()
