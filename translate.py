"""Bidirectional translation between Anthropic Messages API and OpenAI Chat Completions.

Used by the local GPU fallback path to convert incoming Anthropic-format
requests (from Claude Code / openclaw) into OpenAI format for the local
llama.cpp server, and translate responses back.
"""
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("translate")


# ── Anthropic → OpenAI request ────────────────────────────────────────

def anthropic_to_openai_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic Messages API request to OpenAI Chat Completions format."""
    out: Dict[str, Any] = {}

    # System prompt
    system = payload.get("system")
    if system is not None:
        sys_content = _flatten_content(system) if isinstance(system, list) else str(system)
        out["messages"] = [{"role": "system", "content": sys_content}]
    else:
        out["messages"] = []

    # Messages
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            out["messages"].append(_convert_assistant_msg(msg, content))
        elif role == "user":
            converted = _convert_user_msg(msg, content)
            if isinstance(converted, list):
                out["messages"].extend(converted)
            else:
                out["messages"].append(converted)
        else:
            out["messages"].append({"role": role, "content": _flatten_content(content)})

    # Tools
    tools = payload.get("tools")
    if tools:
        out["tools"] = [_convert_tool(t) for t in tools]

    # Passthrough fields
    for key in ("max_tokens", "temperature", "top_p", "stop"):
        if key in payload:
            out[key] = payload[key]

    out["stream"] = False
    return out


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)


def _convert_assistant_msg(msg: Dict, content: Any) -> Dict:
    """Convert an Anthropic assistant message (may contain tool_use blocks)."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    if not isinstance(content, list):
        return {"role": "assistant", "content": _flatten_content(content)}

    text_parts = []
    tool_calls = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    out: Dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        out["tool_calls"] = tool_calls
    # "" not None — llama.cpp rejects assistant messages with null content
    # alongside tool_calls (400), which failed every local GPU fallback.
    out["content"] = " ".join(text_parts) if text_parts else ""
    return out


def _convert_user_msg(msg: Dict, content: Any) -> Dict:
    """Convert an Anthropic user message (may contain images, tool_results)."""
    if isinstance(content, str):
        return {"role": "user", "content": content}

    if not isinstance(content, list):
        return {"role": "user", "content": _flatten_content(content)}

    parts: List[Any] = []
    tool_results = []

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            if src.get("type") == "base64":
                media_type = src.get("media_type", "image/png")
                data_url = f"data:{media_type};base64,{src.get('data', '')}"
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
        elif btype == "tool_result":
            tool_results.append(block)

    if tool_results:
        # Each tool_result becomes a separate tool-role message
        messages = []
        if parts:
            # Emit text content first as a user message
            text = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
            if text:
                messages.append({"role": "user", "content": text})
        for tr in tool_results:
            tr_content = tr.get("content", "")
            tr_text = _flatten_content(tr_content) if isinstance(tr_content, (list, str)) else str(tr_content)
            messages.append({
                "role": "tool",
                "tool_call_id": tr.get("tool_use_id", ""),
                "content": tr_text,
            })
        return messages  # type: ignore[return-value]

    return {"role": "user", "content": parts if parts else ""}


def _convert_tool(tool: Dict) -> Dict:
    schema = tool.get("input_schema", {})
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": schema,
        },
    }


# ── OpenAI → Anthropic response ───────────────────────────────────────

def openai_to_anthropic_response(
    oai_resp: Dict[str, Any],
    original_model: str,
) -> Dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages API format."""
    choice = (oai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")

    # Content blocks
    content: List[Dict[str, Any]] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        raw_args = fn.get("arguments", "{}")
        try:
            inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            inp = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"tu_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "input": inp,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    # Stop reason
    has_tools = any(c.get("type") == "tool_use" for c in content)
    if has_tools:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Usage
    usage = oai_resp.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
