"""
Ollama adapter for Blender extension.

This module provides a compatibility layer that mimics the llama_manager API
but uses Ollama subprocess instead of llama-cpp-python.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .ollama_subprocess import get_ollama


def chat_completion(
    model_path: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 8192,
    tools: Optional[List[Dict]] = None,
    **kwargs,
) -> Dict:
    """Chat completion using Ollama API.

    Args:
        model_path: Ollama model name (e.g. "qwen2.5-coder:7b")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tools: Optional list of tool definitions
        **kwargs: Additional parameters

    Returns:
        Dict with 'content' or 'error' key
    """
    ollama = get_ollama()

    if not ollama.is_running():
        return {"error": "Ollama server is not running"}

    # Use model name directly (it's already an Ollama model name, not a path)
    model_name = model_path

    # Sanitize messages for better model compatibility
    # Many models get confused by 'tool' or 'system' roles appearing mid-chat,
    # or if 'tool' messages lack proper tool_call_ids.
    # We convert them to labeled User messages to ensure the LLM sees them as external inputs.
    sanitized_messages = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system" and i > 0:
            # Mid-chat system messages (e.g. Scene Updates) -> User message with header
            sanitized_messages.append({
                "role": "user",
                "content": f"[System Update]\n{content}"
            })
        elif role == "tool":
            # Tool outputs -> User message with header
            # This is safer than relying on strict tool_call_id matching which might be missing
            name = msg.get("name", "tool")
            sanitized_messages.append({
                "role": "user",
                "content": f"[Tool Output: {name}]\n{content}"
            })
        else:
            sanitized_messages.append(msg)
            
    # Use sanitized messages instead of raw messages
    messages = sanitized_messages

    # Extract Ollama-specific parameters from kwargs (passed from assistant.py)
    # Use conservative defaults to avoid server-side 400 errors and long startups
    n_ctx = kwargs.get("n_ctx", 8192)
    n_gpu_layers = kwargs.get("n_gpu_layers", -1)
    n_batch = kwargs.get("n_batch", 256)

    # Extract debug mode
    debug_mode = kwargs.get("debug_mode", False)

    try:
        url = f"{ollama.base_url}/api/chat"

        # Build options dict from kwargs (no bpy access from background thread)
        # Convert 999 (all layers) to -1 for Ollama
        num_gpu = -1 if n_gpu_layers >= 999 else n_gpu_layers

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_gpu": num_gpu,
            "num_ctx": n_ctx,
            "num_batch": n_batch,
        }

        # Add optional parameters if provided in kwargs
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs and kwargs["top_k"] is not None:
            options["top_k"] = kwargs["top_k"]
        if "repeat_penalty" in kwargs and kwargs["repeat_penalty"] is not None:
            options["repeat_penalty"] = kwargs["repeat_penalty"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            options["seed"] = kwargs["seed"]

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": options,
        }

        # Add keep_alive if provided
        if "keep_alive" in kwargs:
            payload["keep_alive"] = kwargs["keep_alive"]

        # Optional: native tools for supported models
        if tools:
            payload["tools"] = tools

        # Optional: structured outputs (JSON or JSON Schema)
        fmt = kwargs.get("format")
        if fmt is not None:
            payload["format"] = fmt

        # Optional: Ollama 'think' capability (pass-through from assistant)
        if "think" in kwargs and kwargs["think"] is not None:
            payload["think"] = kwargs["think"]

        def _do_request(payload_obj: dict) -> Dict:
            data_local = json.dumps(payload_obj).encode("utf-8")

            req_local = urllib.request.Request(url, data=data_local)

            req_local.add_header("Content-Type", "application/json")

            print(f"[Ollama Adapter] Sending request to {url} with model {model_name}")
            print(
                f"[Ollama Adapter] Payload size: {len(data_local)} bytes, {len(messages)} messages"
            )
            
            if debug_mode:
                print(f"[Ollama Adapter] FULL PAYLOAD:\n{json.dumps(payload_obj, indent=2)}")

            # Always stream; accumulate content and tool_calls
            with urllib.request.urlopen(req_local, timeout=300) as response_local:
                print(f"[Ollama Adapter] Streaming response...")
                content_parts: list[str] = []
                thinking_parts: list[str] = []
                final_tool_calls = None
                last_message = None
                top_level_tool_calls = None

                for raw_line in response_local:
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # Ignore non-JSON keep-alives
                        continue

                    # Debug: Show raw JSON to diagnose thinking fields
                    if debug_mode and not obj.get("done"):
                        print(f"[Ollama Adapter] Chunk: {json.dumps(obj)}")

                    # Accumulate message/content
                    msg = obj.get("message")
                    if isinstance(msg, dict):
                        last_message = msg
                        delta = msg.get("content")
                        if isinstance(delta, str) and delta:
                            content_parts.append(delta)

                        # Capture thinking/reasoning content
                        thinking = (
                            msg.get("thinking")
                            or msg.get("reasoning")
                            or msg.get("thinking_content")
                        )
                        if isinstance(thinking, str) and thinking:
                            thinking_parts.append(thinking)

                        if isinstance(msg.get("tool_calls"), list) and msg.get(
                            "tool_calls"
                        ):
                            final_tool_calls = msg.get("tool_calls")

                    # Some servers may emit tool_calls at top-level
                    if isinstance(obj.get("tool_calls"), list) and obj.get(
                        "tool_calls"
                    ):
                        top_level_tool_calls = obj.get("tool_calls")

                    if obj.get("done") is True:
                        break

                # Build final message
                content = "".join(content_parts) if content_parts else ""
                thinking = "".join(thinking_parts) if thinking_parts else ""
                tool_calls = final_tool_calls or top_level_tool_calls or None

                final_message = {
                    "role": "assistant",
                    "content": content,
                }
                if thinking:
                    final_message["thinking"] = thinking
                if tool_calls:
                    final_message["tool_calls"] = tool_calls

                print(
                    f"[Ollama Adapter] Stream complete: {len(content)} chars, {len(thinking)} thinking chars, {len(tool_calls) if tool_calls else 0} tool calls"
                )
                
                if debug_mode:
                    print(f"[Ollama Adapter] FINAL RESPONSE:\n{json.dumps(final_message, indent=2)}")
                    
                return {"message": final_message}

        try:
            result = _do_request(payload)
        except urllib.error.HTTPError as he:
            # Some models (e.g., Gemma) don't support native tools. Retry once without tools on 400.
            # Also, some models don't support 'think' and will 400; retry once without 'think'.
            body = None
            try:
                body = he.read().decode("utf-8", errors="ignore")
            except Exception:
                body = None
            if he.code == 400 and tools:
                print("[Ollama Adapter] HTTP 400 with tools; retrying without tools...")
                payload_no_tools = dict(payload)
                payload_no_tools.pop("tools", None)
                try:
                    result = _do_request(payload_no_tools)
                except urllib.error.HTTPError as he2:
                    # If still 400 and likely due to 'think', retry without 'think'
                    body2 = None
                    try:
                        body2 = he2.read().decode("utf-8", errors="ignore")
                    except Exception:
                        body2 = None
                    if he2.code == 400 and (
                        ("think" in payload_no_tools)
                        or (
                            body2
                            and (
                                "think" in body2.lower() or "thinking" in body2.lower()
                            )
                        )
                    ):
                        print(
                            "[Ollama Adapter] HTTP 400 with think; retrying without think..."
                        )
                        payload_no_think = dict(payload_no_tools)
                        payload_no_think.pop("think", None)
                        result = _do_request(payload_no_think)
                    else:
                        return {
                            "error": f"Ollama request failed: HTTP Error {he2.code}: {body2 or he2.reason}"
                        }
            elif he.code == 400 and (
                ("think" in payload)
                or (body and ("think" in body.lower() or "thinking" in body.lower()))
            ):
                print("[Ollama Adapter] HTTP 400 with think; retrying without think...")
                payload_no_think = dict(payload)
                payload_no_think.pop("think", None)
                result = _do_request(payload_no_think)
            else:
                return {
                    "error": f"Ollama request failed: HTTP Error {he.code}: {body or he.reason}"
                }

        # Extract content from Ollama response and return message too for native tool-calls
        if "message" in result:
            content = result["message"].get("content", "")
            tool_calls = result["message"].get("tool_calls", [])

            print(
                f"[Ollama Adapter] Got content: {len(content)} chars, {len(tool_calls) if tool_calls else 0} tool calls"
            )

            return {
                "message": result.get("message", {}),
                "content": content,
                "tool_calls": tool_calls if tool_calls else None,
            }

        return {"error": "Invalid response from Ollama"}

    except urllib.error.URLError as e:
        return {"error": f"Ollama request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Chat completion failed: {str(e)}"}


def generate_embedding(model_path: str, text: str, **kwargs) -> Optional[List[float]]:
    """Generate embeddings using Ollama.

    Args:
        model_path: Ollama model name (e.g. "nomic-embed-text")
        text: Text to embed

    Returns:
        List of floats (embedding vector) or None on failure
    """
    ollama = get_ollama()

    if not ollama.is_running():
        print("[Ollama Adapter] Server not running")
        return None

    # Use model name directly
    model_name = model_path

    try:
        # Default to 5m for embeddings if not specified
        keep_alive = kwargs.get("keep_alive", "5m")
        return ollama.generate_embedding(model_name, text, keep_alive=keep_alive)
    except Exception as e:
        print(f"[Ollama Adapter] Embedding failed: {e}")
        return None


# ModelManager class removed - Ollama manages models internally
# Use ollama_subprocess.get_ollama().list_models() to get available models

# Compatibility flag
LLAMA_CPP_AVAILABLE = True  # We're using Ollama, so this is "available"
