"""
tool_executor.py — Executes tools based on classified intents.

All file operations are sandboxed to the output/ directory.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    Intent,
    OUTPUT_DIR,
    CODE_GENERATION_PROMPT,
    SUMMARIZE_PROMPT,
    CHAT_PROMPT,
)
from intent_classifier import call_llm


@dataclass
class ExecutionResult:
    """Result of a tool execution."""
    success: bool
    intent: Intent
    action_description: str
    output_content: str
    file_path: Optional[str] = None
    duration: float = 0.0
    error: Optional[str] = None
    backend: str = ""


def _safe_path(filename: str) -> Path:
    """
    Resolve a filename to a safe path within OUTPUT_DIR.
    Raises ValueError if the path escapes the sandbox.
    
    Args:
        filename: The desired filename (may include subdirectories).
        
    Returns:
        Resolved Path object within OUTPUT_DIR.
    """
    # Strip leading slashes/dots to prevent traversal
    clean = filename.strip().lstrip("/\\").replace("..", "")
    target = (OUTPUT_DIR / clean).resolve()

    # Verify the resolved path is within OUTPUT_DIR
    if not str(target).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError(
            f"Path traversal detected: '{filename}' resolves outside the output directory."
        )

    return target


def _infer_filename(parameters: dict, default_ext: str = ".txt") -> str:
    """Infer a filename from intent parameters."""
    filename = parameters.get("filename")
    if filename and filename != "null":
        return filename

    # Build from description
    desc = parameters.get("description", "file")
    # Create a slugified name
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in desc[:40]).strip("_")
    slug = slug.lower() or "untitled"

    lang = parameters.get("language", "").lower() if parameters.get("language") else ""
    ext_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "c": ".c",
        "cpp": ".cpp",
        "c++": ".cpp",
        "go": ".go",
        "rust": ".rs",
        "html": ".html",
        "css": ".css",
        "sql": ".sql",
        "bash": ".sh",
        "shell": ".sh",
        "ruby": ".rb",
        "php": ".php",
    }
    ext = ext_map.get(lang, default_ext)

    return f"{slug}{ext}"


def execute_create_file(parameters: dict) -> ExecutionResult:
    """
    Create an empty file or directory in the output folder.
    
    Args:
        parameters: Intent parameters with 'filename' and 'description'.
        
    Returns:
        ExecutionResult with success status and file path.
    """
    start = time.time()
    try:
        filename = _infer_filename(parameters, default_ext=".txt")
        target = _safe_path(filename)

        # Check if it looks like a directory request
        is_dir = (
            filename.endswith("/")
            or filename.endswith("\\")
            or "folder" in parameters.get("description", "").lower()
            or "directory" in parameters.get("description", "").lower()
        )

        if is_dir:
            target.mkdir(parents=True, exist_ok=True)
            action = f"Created directory: {target.relative_to(OUTPUT_DIR)}"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
            action = f"Created file: {target.relative_to(OUTPUT_DIR)}"

        return ExecutionResult(
            success=True,
            intent=Intent.CREATE_FILE,
            action_description=action,
            output_content=f"✅ {action}\n📂 Location: {target}",
            file_path=str(target),
            duration=round(time.time() - start, 2),
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            intent=Intent.CREATE_FILE,
            action_description="Failed to create file",
            output_content="",
            duration=round(time.time() - start, 2),
            error=str(e),
        )


def execute_write_code(parameters: dict, groq_api_key: str = None, preference: str = "Auto") -> ExecutionResult:
    """
    Generate code via LLM and save it to a file in the output folder.
    
    Args:
        parameters: Intent parameters with 'description', 'language', 'filename'.
        groq_api_key: Optional Groq API key for fallback.
        preference: User's chosen LLM backend override.
        
    Returns:
        ExecutionResult with generated code and file path.
    """
    start = time.time()
    try:
        description = parameters.get("description", "Write a hello world program")
        language = parameters.get("language", "python") or "python"
        filename = _infer_filename(parameters, default_ext=".py")

        # Generate code via LLM
        prompt = f"Language: {language}\nTask: {description}\n\nGenerate the code:"
        llm_result = call_llm(prompt, CODE_GENERATION_PROMPT, groq_api_key, preference)

        if not llm_result["success"]:
            return ExecutionResult(
                success=False,
                intent=Intent.WRITE_CODE,
                action_description="Failed to generate code",
                output_content="",
                duration=round(time.time() - start, 2),
                error=llm_result["error"],
                backend=llm_result["backend"],
            )

        code = llm_result["response"].strip()

        # Remove markdown code fences if present
        if code.startswith("```"):
            lines = code.split("\n")
            # Remove first line (```language)
            lines = lines[1:]
            # Remove last line if it's ``
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        # Write to file
        target = _safe_path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")

        return ExecutionResult(
            success=True,
            intent=Intent.WRITE_CODE,
            action_description=f"Generated {language} code → {target.relative_to(OUTPUT_DIR)}",
            output_content=code,
            file_path=str(target),
            duration=round(time.time() - start, 2),
            backend=llm_result["backend"],
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            intent=Intent.WRITE_CODE,
            action_description="Failed to write code",
            output_content="",
            duration=round(time.time() - start, 2),
            error=str(e),
        )


def execute_summarize(parameters: dict, transcript: str, groq_api_key: str = None, preference: str = "Auto") -> ExecutionResult:
    """
    Summarize text content via LLM.
    
    Args:
        parameters: Intent parameters with 'content' or 'description'.
        transcript: Original transcript (used as content if no explicit content).
        groq_api_key: Optional Groq API key for fallback.
        preference: User's chosen LLM backend override.
        
    Returns:
        ExecutionResult with the summary.
    """
    start = time.time()
    try:
        content = parameters.get("content") or parameters.get("description") or transcript
        if content in (None, "null", ""):
            content = transcript

        prompt = f"Please summarize the following:\n\n{content}"
        llm_result = call_llm(prompt, SUMMARIZE_PROMPT, groq_api_key, preference)

        if not llm_result["success"]:
            return ExecutionResult(
                success=False,
                intent=Intent.SUMMARIZE,
                action_description="Failed to summarize text",
                output_content="",
                duration=round(time.time() - start, 2),
                error=llm_result["error"],
                backend=llm_result["backend"],
            )

        return ExecutionResult(
            success=True,
            intent=Intent.SUMMARIZE,
            action_description="Summarized the provided text",
            output_content=llm_result["response"].strip(),
            duration=round(time.time() - start, 2),
            backend=llm_result["backend"],
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            intent=Intent.SUMMARIZE,
            action_description="Failed to summarize",
            output_content="",
            duration=round(time.time() - start, 2),
            error=str(e),
        )


def execute_general_chat(transcript: str, groq_api_key: str = None, preference: str = "Auto") -> ExecutionResult:
    """
    Handle a general chat / conversational query via LLM.
    
    Args:
        transcript: The user's transcribed speech.
        groq_api_key: Optional Groq API key for fallback.
        preference: User's chosen LLM backend override.
        
    Returns:
        ExecutionResult with the chat response.
    """
    start = time.time()
    try:
        llm_result = call_llm(transcript, CHAT_PROMPT, groq_api_key, preference)

        if not llm_result["success"]:
            return ExecutionResult(
                success=False,
                intent=Intent.GENERAL_CHAT,
                action_description="Failed to get chat response",
                output_content="",
                duration=round(time.time() - start, 2),
                error=llm_result["error"],
                backend=llm_result["backend"],
            )

        return ExecutionResult(
            success=True,
            intent=Intent.GENERAL_CHAT,
            action_description="Generated conversational response",
            output_content=llm_result["response"].strip(),
            duration=round(time.time() - start, 2),
            backend=llm_result["backend"],
        )

    except Exception as e:
        return ExecutionResult(
            success=False,
            intent=Intent.GENERAL_CHAT,
            action_description="Failed to respond",
            output_content="",
            duration=round(time.time() - start, 2),
            error=str(e),
        )


def execute_tool(
    intent: Intent,
    parameters: dict,
    transcript: str,
    groq_api_key: str = None,
    preference: str = "Auto",
) -> ExecutionResult:
    """
    Dispatch to the correct tool handler based on the classified intent.
    
    Args:
        intent: The classified intent enum.
        parameters: Extracted parameters from intent classification.
        transcript: The original transcribed text.
        groq_api_key: Optional Groq API key.
        preference: Chosen LLM backend.
        
    Returns:
        ExecutionResult from the appropriate tool.
    """
    dispatch = {
        Intent.CREATE_FILE: lambda: execute_create_file(parameters),
        Intent.WRITE_CODE: lambda: execute_write_code(parameters, groq_api_key, preference),
        Intent.SUMMARIZE: lambda: execute_summarize(parameters, transcript, groq_api_key, preference),
        Intent.GENERAL_CHAT: lambda: execute_general_chat(transcript, groq_api_key, preference),
    }

    handler = dispatch.get(intent, lambda: execute_general_chat(transcript, groq_api_key, preference))
    return handler()
