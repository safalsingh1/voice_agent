"""
intent_classifier.py — Intent classification using Ollama (local) or Groq (fallback).
"""

import json
import time
import requests

from config import (
    Intent,
    INTENT_SYSTEM_PROMPT,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    GROQ_MODEL,
)


def _call_ollama(prompt: str, system_prompt: str) -> str:
    """
    Call the local Ollama API for chat completion.
    
    Args:
        prompt: The user message.
        system_prompt: The system instruction.
        
    Returns:
        The model's response text.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_groq(prompt: str, system_prompt: str, groq_api_key: str) -> str:
    """
    Call Groq API for chat completion (fallback).
    
    Args:
        prompt: The user message.
        system_prompt: The system instruction.
        groq_api_key: Groq API key.
        
    Returns:
        The model's response text.
    """
    from groq import Groq

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_llm(prompt: str, system_prompt: str, groq_api_key: str = None, preference: str = "Auto") -> dict:
    """
    Call LLM based on user preference (Auto, Ollama Only, Groq API Only).
    
    Args:
        prompt: The user message.
        system_prompt: The system instruction.
        groq_api_key: Optional Groq API key for fallback.
        preference: "Auto", "Ollama Only", or "Groq API Only".
        
    Returns:
        dict with keys: response, backend, duration, success, error
    """
    start_time = time.time()
    ollama_error = "Ollama skipped by user preference"

    # Try Ollama (unless Groq is strictly preferred)
    if preference in ("Auto", "Ollama Only"):
        if _check_ollama_available():
            try:
                response = _call_ollama(prompt, system_prompt)
                return {
                    "response": response,
                    "backend": f"Ollama ({OLLAMA_MODEL})",
                    "duration": round(time.time() - start_time, 2),
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                ollama_error = str(e)
                if preference == "Ollama Only":
                    return {
                        "response": "", "backend": "none", "duration": round(time.time() - start_time, 2),
                        "success": False, "error": f"Ollama: {ollama_error}"
                    }
        else:
            ollama_error = "Ollama not available"
            if preference == "Ollama Only":
                return {
                    "response": "", "backend": "none", "duration": round(time.time() - start_time, 2),
                    "success": False, "error": f"Ollama: {ollama_error}"
                }

    # Try Groq (Auto fallback or strict preference)
    if preference in ("Auto", "Groq API Only"):
        if groq_api_key:
            try:
                response = _call_groq(prompt, system_prompt, groq_api_key)
                return {
                    "response": response,
                    "backend": f"Groq ({GROQ_MODEL})",
                    "duration": round(time.time() - start_time, 2),
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "response": "",
                    "backend": "none",
                    "duration": round(time.time() - start_time, 2),
                    "success": False,
                    "error": f"Groq Error: {str(e)}" if preference == "Groq API Only" else f"Ollama: {ollama_error} | Groq: {str(e)}",
                }
        elif preference == "Groq API Only":
            return {
                "response": "", "backend": "none", "duration": round(time.time() - start_time, 2),
                "success": False, "error": "Groq API Only selected, but no API key provided.",
            }

    return {
        "response": "",
        "backend": "none",
        "duration": round(time.time() - start_time, 2),
        "success": False,
        "error": f"Ollama: {ollama_error} | Groq: No API key provided",
    }


def classify_intent(transcript: str, groq_api_key: str = None, preference: str = "Auto") -> dict:
    """
    Classify the intent of a transcribed voice command.
    
    Args:
        transcript: The transcribed text from speech.
        groq_api_key: Optional Groq API key for fallback.
        preference: "Auto", "Ollama Only", or "Groq API Only".
        
    Returns:
        dict with keys: intent (Intent enum), parameters (dict),
                        raw_response, backend, duration, success, error
    """
    llm_result = call_llm(transcript, INTENT_SYSTEM_PROMPT, groq_api_key, preference)

    if not llm_result["success"]:
        return {
            "actions": [{"intent": Intent.GENERAL_CHAT, "parameters": {"description": transcript}}],
            "raw_response": "",
            "backend": llm_result["backend"],
            "duration": llm_result["duration"],
            "success": False,
            "error": llm_result["error"],
        }

    # Parse the JSON response
    raw = llm_result["response"].strip()

    # Try to extract JSON from the response (handle markdown fences)
    json_str = raw
    if "```" in raw:
        # Extract content between code fences
        parts = raw.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                json_str = stripped
                break

    try:
        parsed = json.loads(json_str)
        
        # Handle new compound array format or old single-object format
        if "actions" in parsed and isinstance(parsed["actions"], list):
            actions_list = parsed["actions"]
        elif "intent" in parsed:
            actions_list = [parsed]
        else:
            actions_list = [{"intent": "general_chat", "parameters": {"description": transcript}}]

        processed_actions = []
        for a in actions_list:
            intent = Intent.from_string(a.get("intent", "general_chat"))
            parameters = a.get("parameters", {"description": transcript})
            processed_actions.append({"intent": intent, "parameters": parameters})

        return {
            "actions": processed_actions,
            "raw_response": raw,
            "backend": llm_result["backend"],
            "duration": llm_result["duration"],
            "success": True,
            "error": None,
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        # Graceful degradation: default to general_chat
        return {
            "actions": [{"intent": Intent.GENERAL_CHAT, "parameters": {"description": transcript}}],
            "raw_response": raw,
            "backend": llm_result["backend"],
            "duration": llm_result["duration"],
            "success": True,
            "error": f"JSON parse fallback: {str(e)}",
        }
