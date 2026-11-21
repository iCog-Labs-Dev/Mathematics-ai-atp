"""Spawned source-row decoding with one requested row and bounded cleanup.

The child owns blocking dataset setup/downloads. It never receives a model,
Lean handle or client. Spawning is safe after the parent has initialized CUDA.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import multiprocessing


@dataclass(frozen=True)
class AdmissionSource:
    kind: str
    parameters: dict


def _decode_rows(source, connection):
    from .seed_admission import AdmissionReport, _dataset_row_events, _file_row_events

    report = AdmissionReport()
    if source.kind == "dataset":
        rows = _dataset_row_events(**source.parameters, report=report)
    elif source.kind == "file":
        rows = _file_row_events(**source.parameters, report=report)
    elif source.kind == "rows":
        rows = iter(source.parameters["rows"])
    else:
        raise ValueError(f"Unknown admission source {source.kind}")
    try:
        while connection.recv() == "next":
            try:
                event = next(rows, None)
                connection.send((event, report, None))
                if event is None:
                    break
            except Exception as exc:
                connection.send((None, report, f"{type(exc).__name__}: {exc}"))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class SourceRows:
    def __init__(self, source: AdmissionSource):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_decode_rows, args=(source, child), daemon=True
        )
        self.process.start()
        child.close()
        self.pending = None

    async def next(self, deadline):
        if self.pending is None:
            self.connection.send("next")
            self.pending = asyncio.create_task(asyncio.to_thread(self.connection.recv))
        async with asyncio.timeout_at(deadline):
            result = await asyncio.shield(self.pending)
        self.pending = None
        event, report, error = result
        if error is not None:
            raise RuntimeError(f"Dataset source failed: {error}")
        return event, report

    async def close(self):
        if self.process.is_alive():
            self.process.terminate()
        await asyncio.to_thread(self.process.join, 1.0)
        if self.process.is_alive():
            self.process.kill()
            await asyncio.to_thread(self.process.join, 3.0)
        if self.process.is_alive():
            raise RuntimeError("Admission source did not terminate")
        self.connection.close()
        if self.pending is not None:
            await asyncio.gather(self.pending, return_exceptions=True)
            self.pending = None
        self.process.close()
