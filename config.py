"""
config.py — Central configuration for the Voice-Controlled AI Agent.
"""

import os
from enum import Enum
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Whisper STT Configuration ────────────────────────────────────────────────
WHISPER_MODELS = ["base", "small", "medium"]
DEFAULT_WHISPER_MODEL = "small"
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"]

# ─── LLM Configuration ───────────────────────────────────────────────────────
OLLAMA_MODEL = "gemma4:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Intent Enum ──────────────────────────────────────────────────────────────
class Intent(str, Enum):
    CREATE_FILE = "create_file"
    WRITE_CODE = "write_code"
    SUMMARIZE = "summarize"
    GENERAL_CHAT = "general_chat"

    @classmethod
    def from_string(cls, value: str) -> "Intent":
        """Safely parse an intent string, defaulting to GENERAL_CHAT."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            return cls.GENERAL_CHAT

INTENT_LABELS = {
    Intent.CREATE_FILE: ("📁", "Create File", "#4CAF50"),
    Intent.WRITE_CODE: ("💻", "Write Code", "#2196F3"),
    Intent.SUMMARIZE: ("📝", "Summarize", "#FF9800"),
    Intent.GENERAL_CHAT: ("💬", "General Chat", "#9C27B0"),
}

# ─── System Prompts ──────────────────────────────────────────────────────────
INTENT_SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent. Analyze the user's transcribed speech and classify it into one OR MORE of these intents:

1. "create_file" — The user wants to create an empty file or folder (no code content).
2. "write_code" — The user wants to generate code and save it to a file.
3. "summarize" — The user wants to summarize some text or content.
4. "general_chat" — General question, greeting, or anything that doesn't fit the above.

Respond with ONLY valid JSON in this exact format (an array of actions):
{
  "actions": [
    {
      "intent": "create_file | write_code | summarize | general_chat",
      "parameters": {
        "filename": "suggested filename if applicable, or null",
        "language": "programming language if code-related, or null",
        "content": "text to summarize if applicable, or null",
        "description": "brief description of what the user wants for this specific action"
      }
    }
  ]
}

Rules:
- If the user has multiple requests (compound commands, e.g., "Summarize this and save it to a file"), you MUST split them into multiple sequential actions in the "actions" array.
- "create_file" is ONLY for creating blank/empty files or directories.
- If unsure, default to "general_chat".
- Always suggest a reasonable filename if one isn't specified.
- If the user references previous code/actions, use the provided CONVERSATION HISTORY to inform the new parameters and description.
"""

CODE_GENERATION_PROMPT = """You are an expert code generator. Generate clean, well-commented, production-ready code based on the user's request.

Rules:
- Output ONLY the code, no markdown fences, no explanations before or after.
- Include proper docstrings and comments.
- Follow best practices for the specified language.
- If no language is specified, default to Python.
- If a CONVERSATION HISTORY is provided, you MUST modify the previously generated code according to the new instructions rather than writing from scratch.
"""

SUMMARIZE_PROMPT = """You are a text summarizer. Provide a clear, concise summary of the given text.
Keep the summary informative but significantly shorter than the original.
Maintain the key points and important details."""

CHAT_PROMPT = """You are a helpful, friendly AI assistant. You are part of a voice-controlled agent system.
Provide clear, concise, and helpful responses to the user's questions or statements.
Be conversational but informative.
CRITICAL RULE: Always respond in English unless the user explicitly asks you to speak another language. Do NOT write in Urdu or any other language just because the previous conversation history contained it."""
