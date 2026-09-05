"""Synthesize Anthropic Messages API SSE stream from a complete response."""
import json
import uuid
from typing import Any, Dict, Iterator


def synthesize(response: Dict[str, Any]) -> Iterator[bytes]:
    """Yield SSE event bytes from a complete Anthropic Messages API response."""
    msg_id = response.get("id", f"msg_{uuid.uuid4().hex[:24]}")
    model = response.get("model", "unknown")
    role = response.get("role", "assistant")
    content_blocks = response.get("content", [])
    stop_reason = response.get("stop_reason", "end_turn")
    usage = response.get("usage", {})

    # message_start
    yield _event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": role,
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
        },
    })

    # Content blocks
    for idx, block in enumerate(content_blocks):
        block_type = block.get("type", "text")

        # content_block_start
        if block_type == "text":
            yield _event("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            })

            # content_block_delta — chunk text for natural streaming
            text = block.get("text", "")
            for chunk in _chunk_text(text, 30):
                yield _event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": chunk},
                })

        elif block_type == "tool_use":
            yield _event("content_block_start", {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": block.get("name", ""),
                    "input": {},
                },
            })

            # Send tool input as a single delta
            input_json = json.dumps(block.get("input", {}))
            yield _event("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": input_json},
            })

        # content_block_stop
        yield _event("content_block_stop", {
            "type": "content_block_stop",
            "index": idx,
        })

    # message_delta
    yield _event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    })

    # message_stop
    yield _event("message_stop", {"type": "message_stop"})


def _event(event_type: str, data: Dict[str, Any]) -> bytes:
    """Format a single SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def _chunk_text(text: str, size: int) -> Iterator[str]:
    """Split text into chunks of approximately `size` characters."""
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i:i + size]
