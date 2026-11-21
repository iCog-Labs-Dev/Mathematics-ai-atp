"""Async wrappers that retain ownership until blocking work actually exits."""
from __future__ import annotations

import asyncio


async def owned_thread(function, *args, **kwargs):
    """Cancellation waits for the thread; cancelling its Future cannot stop it."""
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise
