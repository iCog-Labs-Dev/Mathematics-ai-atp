"""Private, bounded IPC between the API process and its runtime process.

Messages are JSON objects framed by a 4-byte big-endian length over a Unix
socket in a directory only the service user can open. Every message names
its ``type`` and the runtime ``generation`` it belongs to; a reader drops
messages from another generation, so a restarted runtime cannot answer for
work its predecessor owned. No model, tensor or Lean handle crosses: only
plain data that ``json`` can encode.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

HEADER = struct.Struct(">I")


class IPCError(RuntimeError):
    """The peer sent an invalid or oversized message, or went away."""


async def send_message(writer: asyncio.StreamWriter, message: dict[str, Any], *, limit: int) -> None:
    body = json.dumps(message, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(body) > limit:
        raise IPCError(f"Message of {len(body)} bytes exceeds the {limit}-byte limit")
    writer.write(HEADER.pack(len(body)) + body)
    await writer.drain()


async def read_message(reader: asyncio.StreamReader, *, limit: int) -> dict[str, Any]:
    try:
        header = await reader.readexactly(HEADER.size)
    except asyncio.IncompleteReadError as exc:
        raise IPCError("The peer closed the connection") from exc
    (size,) = HEADER.unpack(header)
    if size > limit:
        raise IPCError(f"Peer announced {size} bytes, over the {limit}-byte limit")
    try:
        body = await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise IPCError("The peer closed the connection mid-message") from exc
    try:
        message = json.loads(body)
    except ValueError as exc:
        raise IPCError("The peer sent invalid JSON") from exc
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise IPCError("A message must be an object with a string type")
    if not isinstance(message.get("generation"), int):
        raise IPCError("A message must name its runtime generation")
    return message
