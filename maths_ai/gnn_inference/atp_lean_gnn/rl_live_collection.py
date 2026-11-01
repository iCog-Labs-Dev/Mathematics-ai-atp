"""Lifecycle and bounded concurrency for live RL theorem collection."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Sequence

import torch

from maths_ai.lean_replay import ReplayInputError
from pantograph_async import (
    await_cleanup,
    PantographConfigurationError,
    PantographProtocolError,
    PantographRequestTimeout,
    PantographStateError,
    PantographTransportError,
)

from .rl_resources import ResourcePlan


# "inadmissible": the seed could not be replayed faithfully, so no search ran.
CollectionOutcome = Literal[
    "solved",
    "unsolved",
    "cooperative_deadline",
    "hard_timeout",
    "error",
    "inadmissible",
]
ServerFactory = Callable[[int], Awaitable[Any]]
ReasonerFactory = Callable[[int, torch.device, torch.nn.Module, Any], Any]
# (reasoner, theorem, proposal, deadline): the proposal is the item's own, never the worker's.
SearchFactory = Callable[[Any, Any, Any, float], Awaitable[Any]]
SolvedPredicate = Callable[[Any], bool]
CooperativeDeadlinePredicate = Callable[[Any], bool]
LogFunction = Callable[[str], None]

# Startup and search budgets use ``asyncio.timeout`` rather than ``wait_for``:
# the awaited coroutine stays in the caller's task, so a repeated cancellation
# during shutdown cannot detach a client whose cleanup is still running.

# Errors that mean the client contract itself is broken; no retry can help.
FATAL_SESSION_ERRORS = (
    PantographProtocolError,
    PantographStateError,
    PantographConfigurationError,
)


@dataclass
class LiveReasonerWorker:
    """One model binding, reasoner, and Pantograph server."""

    worker_id: int
    device: torch.device
    model: torch.nn.Module
    reasoner: Any
    server: Any
    active: bool = True


@dataclass(frozen=True)
class CollectionRecord:
    """Outcome and timing for one attempted theorem."""

    batch_index: int
    worker_id: int
    queue_wait_s: float
    search_s: float
    outcome: CollectionOutcome
    result: Any | None = None
    exception_category: str | None = None


@dataclass(frozen=True)
class CollectionRound:
    """Ordered successful results, complete attempt records, and summary metrics."""

    results: list[Any]
    records: list[CollectionRecord]
    stats: dict[str, float]


class WorkerStartupError(RuntimeError):
    """A worker could not create and probe its server/reasoner pair."""

    def __init__(
        self,
        *,
        worker_id: int,
        device: torch.device,
        environment_description: str,
        cause: BaseException,
    ) -> None:
        self.worker_id = worker_id
        self.device = device
        self.environment_description = environment_description
        self.cause = cause
        super().__init__(
            f"Pantograph worker {worker_id} on {device} failed to start in "
            f"{environment_description}: {type(cause).__name__}: {cause}"
        )


async def _close_servers(servers: list[Any]) -> None:
    """Close and reap every client; each close is bounded by its own settings."""

    async def close_all():
        results = await asyncio.gather(
            *(server.aclose() for server in servers), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    await await_cleanup(
        asyncio.create_task(close_all(), name="pantograph-pool-close-all")
    )


async def _close_partial_server(server: Any, original: BaseException) -> None:
    try:
        await server.aclose()
    except BaseException as cleanup:
        if isinstance(original, asyncio.CancelledError):
            original.add_note(f"Partial worker cleanup also reported: {cleanup}")
            raise original
        raise


class LiveReasonerPool:
    """One-server/one-reasoner workers sharing read-only per-device replicas."""

    def __init__(
        self,
        *,
        workers: list[LiveReasonerWorker],
        plan: ResourcePlan,
        canonical_model: torch.nn.Module,
        server_factory: ServerFactory,
        reasoner_factory: ReasonerFactory,
        search_factory: SearchFactory,
        is_solved: SolvedPredicate,
        is_cooperative_deadline: CooperativeDeadlinePredicate,
        startup_timeout_s: float,
        shutdown_timeout_s: float,
        queue_size: int,
        environment_description: str,
        model_generation: int,
        log: LogFunction,
    ) -> None:
        self.workers = workers
        self.plan = plan
        self.canonical_model = canonical_model
        self.server_factory = server_factory
        self.reasoner_factory = reasoner_factory
        self.search_factory = search_factory
        self.is_solved = is_solved
        self.is_cooperative_deadline = is_cooperative_deadline
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self.queue_size = max(int(queue_size), 0)
        self.environment_description = environment_description
        self.model_generation = int(model_generation)
        self.log = log
        self._closed = False
        self.close_callbacks = []
        self._close_task: asyncio.Task | None = None
        self._collecting = False
        self._collection_task: asyncio.Task | None = None
        self._replica_generations: dict[int, int] = {
            id(worker.model): self.model_generation for worker in workers
        }

    @classmethod
    async def create(
        cls,
        *,
        model: torch.nn.Module,
        plan: ResourcePlan,
        server_factory: ServerFactory,
        reasoner_factory: ReasonerFactory,
        search_factory: SearchFactory,
        is_solved: SolvedPredicate,
        is_cooperative_deadline: CooperativeDeadlinePredicate,
        startup_timeout_s: float,
        shutdown_timeout_s: float,
        queue_size: int = 0,
        environment_description: str = "Pantograph environment",
        model_generation: int = 0,
        log: LogFunction = print,
    ) -> "LiveReasonerPool":
        """Create every worker or roll back every successful partial startup."""

        replicas: dict[str, torch.nn.Module] = {}
        bindings: list[tuple[int, torch.device, torch.nn.Module]] = []
        for worker_id in range(plan.worker_count):
            device = plan.collection_devices[worker_id % len(plan.collection_devices)]
            key = str(device)
            replica = replicas.get(key)
            if replica is None:
                replica = model if device == plan.update_device else copy.deepcopy(model).to(device)
                replica.eval()
                replicas[key] = replica
            bindings.append((worker_id, device, replica))

        async def create_one(
            worker_id: int,
            device: torch.device,
            replica: torch.nn.Module,
        ) -> LiveReasonerWorker:
            server = None
            try:
                async with asyncio.timeout(startup_timeout_s):
                    server = await server_factory(worker_id)
                reasoner = reasoner_factory(worker_id, device, replica, server)
                return LiveReasonerWorker(worker_id, device, replica, reasoner, server)
            except BaseException as exc:
                if server is not None:
                    await _close_partial_server(server, exc)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise WorkerStartupError(
                    worker_id=worker_id,
                    device=device,
                    environment_description=environment_description,
                    cause=exc,
                ) from exc

        tasks = [asyncio.create_task(create_one(*binding)) for binding in bindings]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException as original:
            # Cancelled during startup: stop the remaining startups, then close
            # every client that finished before handing the cancellation back.

            async def rollback():
                for task in tasks:
                    task.cancel()
                completed = await asyncio.gather(*tasks, return_exceptions=True)
                await _close_servers(
                    [
                        item.server
                        for item in completed
                        if isinstance(item, LiveReasonerWorker)
                    ]
                )

            try:
                await await_cleanup(
                    asyncio.create_task(
                        rollback(), name="pantograph-pool-startup-rollback"
                    )
                )
            except BaseException as cleanup:
                if isinstance(original, asyncio.CancelledError):
                    original.add_note(f"Startup rollback also reported: {cleanup}")
                else:
                    raise
            raise original
        workers = [result for result in results if isinstance(result, LiveReasonerWorker)]
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            await _close_servers([worker.server for worker in workers])
            raise errors[0]

        return cls(
            workers=workers,
            plan=plan,
            canonical_model=model,
            server_factory=server_factory,
            reasoner_factory=reasoner_factory,
            search_factory=search_factory,
            is_solved=is_solved,
            is_cooperative_deadline=is_cooperative_deadline,
            startup_timeout_s=startup_timeout_s,
            shutdown_timeout_s=shutdown_timeout_s,
            queue_size=queue_size,
            environment_description=environment_description,
            model_generation=model_generation,
            log=log,
        )

    async def __aenter__(self) -> "LiveReasonerPool":
        if self._closed:
            raise RuntimeError("Cannot enter a closed LiveReasonerPool.")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    @property
    def primary(self) -> Any:
        for worker in self.workers:
            if worker.active:
                return worker.reasoner
        raise RuntimeError("The live reasoner pool has no active workers.")

    @property
    def active_workers(self) -> list[LiveReasonerWorker]:
        return [worker for worker in self.workers if worker.active]

    async def close(self) -> None:
        """Close every current client once; repeated calls join the same close."""

        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(
                self._shutdown_pool(), name="pantograph-pool-close"
            )
        await await_cleanup(self._close_task)

    async def _shutdown_pool(self) -> None:
        # Join collection before snapshotting clients, so a replacement cannot
        # publish a new process after the shutdown list has been taken.
        if self._collection_task is not None:
            self._collection_task.cancel()
            await asyncio.gather(self._collection_task, return_exceptions=True)
        try:
            await _close_servers([worker.server for worker in self.workers])
        finally:
            for close in reversed(self.close_callbacks):
                await close()

    def assert_generation(self, model_generation: int) -> None:
        """Require every active replica to contain the canonical snapshot."""

        if model_generation != self.model_generation:
            raise RuntimeError(
                f"Pool generation {self.model_generation} does not match canonical "
                f"generation {model_generation}."
            )
        for worker in self.active_workers:
            replica_generation = self._replica_generations.get(id(worker.model))
            if replica_generation != model_generation:
                raise RuntimeError(
                    f"Worker {worker.worker_id} replica generation {replica_generation} "
                    f"does not match canonical generation {model_generation}."
                )
            worker.model.eval()

    def synchronize(self, model: torch.nn.Module, model_generation: int) -> float:
        """Copy the final canonical state to each distinct collection replica."""

        if self._collecting:
            raise PantographStateError(
                "Cannot synchronize model weights during collection."
            )
        if model is not self.canonical_model:
            raise ValueError("Replica synchronization requires the pool's canonical model.")
        if model_generation <= self.model_generation:
            raise ValueError(
                f"New model generation {model_generation} must be greater than "
                f"pool generation {self.model_generation}."
            )
        started_at = time.perf_counter()
        state = model.state_dict()
        seen: set[int] = set()
        for worker in self.active_workers:
            replica = worker.model
            if id(replica) in seen:
                continue
            seen.add(id(replica))
            if replica is not model:
                replica.load_state_dict(state, strict=True)
                replica.to(worker.device)
            replica.eval()
            self._replica_generations[id(replica)] = model_generation
        self.model_generation = model_generation
        return time.perf_counter() - started_at

    async def _replacement_worker(
        self, worker: LiveReasonerWorker
    ) -> LiveReasonerWorker:
        """Retire the complete client/reasoner binding and start a fresh one."""
        if self._closed:
            raise PantographStateError("Cannot replace a worker in a closing pool.")
        await _close_servers([worker.server])
        server = None
        try:
            async with asyncio.timeout(self.startup_timeout_s):
                server = await self.server_factory(worker.worker_id)
            reasoner = self.reasoner_factory(
                worker.worker_id, worker.device, worker.model, server
            )
            replacement = LiveReasonerWorker(
                worker.worker_id,
                worker.device,
                worker.model,
                reasoner,
                server,
            )
            self.workers[worker.worker_id] = replacement
            return replacement
        except BaseException as exc:
            if server is not None:
                await _close_partial_server(server, exc)
            worker.active = False
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise WorkerStartupError(
                worker_id=worker.worker_id,
                device=worker.device,
                environment_description=self.environment_description,
                cause=exc,
            ) from exc

    async def collect_round(
        self,
        batch: list[Any],
        *,
        timeout_s: float,
        model_generation: int,
        proposals: Sequence[Any],
    ) -> CollectionRound:
        """Collect one bounded round and return records in input order.

        ``proposals[i]`` goes with ``batch[i]`` to whichever worker searches it,
        including a replacement worker, so an item's draws do not depend on
        scheduling. The round runs as one task that shutdown can cancel and
        join, and the pool rejects an overlapping round or a weight update
        while it runs.
        """
        if len(proposals) != len(batch):
            raise ValueError(
                f"Expected one proposal per item: {len(proposals)} for {len(batch)} items."
            )
        if self._collecting:
            raise PantographStateError("A pool supports one active collection round.")
        self._collecting = True
        self._collection_task = asyncio.create_task(
            self._collect_round(
                batch,
                timeout_s=timeout_s,
                model_generation=model_generation,
                proposals=proposals,
            ),
            name="pantograph-pool-collection",
        )
        try:
            return await self._collection_task
        finally:
            self._collecting = False
            self._collection_task = None

    async def _collect_round(
        self,
        batch: list[Any],
        *,
        timeout_s: float,
        model_generation: int,
        proposals: Sequence[Any],
    ) -> CollectionRound:

        if self._closed:
            raise RuntimeError("Cannot collect with a closed LiveReasonerPool.")
        self.assert_generation(model_generation)
        workers = self.active_workers
        if not workers:
            raise RuntimeError("The live reasoner pool has no active workers.")

        round_started = time.perf_counter()
        queue: asyncio.Queue[tuple[int, Any, float] | None] = asyncio.Queue(
            maxsize=len(workers) + self.queue_size
        )
        records: list[CollectionRecord] = []
        replacements: dict[int, int] = {worker.worker_id: 0 for worker in workers}
        replacement_failures: dict[int, int] = {worker.worker_id: 0 for worker in workers}
        active_count = len(workers)

        async def produce() -> None:
            for index, theorem in enumerate(batch):
                await queue.put((index, theorem, time.perf_counter()))
            for _ in workers:
                await queue.put(None)

        async def run_worker(initial_worker: LiveReasonerWorker) -> None:
            nonlocal active_count
            worker = initial_worker
            while True:
                item = await queue.get()
                if item is None:
                    return
                index, theorem, enqueued_at = item
                search_started = time.perf_counter()
                queue_wait = search_started - enqueued_at
                backstop = asyncio.timeout(timeout_s * 1.25)
                try:
                    cooperative_deadline = time.monotonic() + timeout_s
                    async with backstop:
                        result = await self.search_factory(
                            worker.reasoner,
                            theorem,
                            proposals[index],
                            cooperative_deadline,
                        )
                    if self.is_cooperative_deadline(result):
                        outcome: CollectionOutcome = "cooperative_deadline"
                    else:
                        outcome = "solved" if self.is_solved(result) else "unsolved"
                    records.append(
                        CollectionRecord(
                            batch_index=index,
                            worker_id=worker.worker_id,
                            queue_wait_s=queue_wait,
                            search_s=time.perf_counter() - search_started,
                            outcome=outcome,
                            result=result,
                        )
                    )
                except ReplayInputError as exc:
                    # A property of the theorem, not of the session: the
                    # client stays usable, so the worker is kept.
                    records.append(
                        CollectionRecord(
                            batch_index=index,
                            worker_id=worker.worker_id,
                            queue_wait_s=queue_wait,
                            search_s=time.perf_counter() - search_started,
                            outcome="inadmissible",
                            exception_category=exc.reason.value,
                        )
                    )
                    self.log(
                        f"  [collect worker={worker.worker_id}] inadmissible seed: {exc}"
                    )
                except (asyncio.TimeoutError, PantographTransportError) as exc:
                    # The interrupted session may hold a request in flight, so
                    # the whole client/reasoner binding is retired.
                    hard_timeout = backstop.expired() or isinstance(
                        exc, (asyncio.TimeoutError, PantographRequestTimeout)
                    )
                    records.append(
                        CollectionRecord(
                            batch_index=index,
                            worker_id=worker.worker_id,
                            queue_wait_s=queue_wait,
                            search_s=time.perf_counter() - search_started,
                            outcome="hard_timeout" if hard_timeout else "error",
                            exception_category=type(exc).__name__,
                        )
                    )
                    self.log(
                        f"  [collect worker={worker.worker_id}] "
                        f"{'hard timeout' if hard_timeout else type(exc).__name__}; "
                        "replacing Pantograph worker"
                    )
                    try:
                        worker = await self._replacement_worker(worker)
                        replacements[worker.worker_id] += 1
                    except WorkerStartupError as replacement_error:
                        if isinstance(replacement_error.cause, FATAL_SESSION_ERRORS):
                            raise replacement_error.cause
                        replacement_failures[worker.worker_id] += 1
                        active_count -= 1
                        self.log(f"  [collect worker={worker.worker_id}] {replacement_error}")
                        if active_count == 0:
                            raise RuntimeError(
                                "Every live collection worker became unavailable."
                            ) from replacement_error
                        return
                except FATAL_SESSION_ERRORS:
                    raise
                except Exception as exc:
                    records.append(
                        CollectionRecord(
                            batch_index=index,
                            worker_id=worker.worker_id,
                            queue_wait_s=queue_wait,
                            search_s=time.perf_counter() - search_started,
                            outcome="error",
                            exception_category=type(exc).__name__,
                        )
                    )
                    self.log(
                        f"  [collect worker={worker.worker_id}] "
                        f"{type(exc).__name__}: {exc}"
                    )

        producer = asyncio.create_task(produce())
        tasks = [asyncio.create_task(run_worker(worker)) for worker in workers]
        try:
            await asyncio.gather(producer, *tasks)
        except BaseException as original:

            async def cancel_and_join():
                producer.cancel()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(producer, *tasks, return_exceptions=True)

            try:
                await await_cleanup(
                    asyncio.create_task(
                        cancel_and_join(), name="pantograph-pool-round-rollback"
                    )
                )
            except BaseException as cleanup:
                if isinstance(original, asyncio.CancelledError):
                    original.add_note(f"Collection rollback also reported: {cleanup}")
                else:
                    raise
            raise original

        records.sort(key=lambda record: record.batch_index)
        successful = [record for record in records if record.result is not None]
        queue_wait_total = sum(record.queue_wait_s for record in records)
        search_total = sum(record.search_s for record in records)
        stats: dict[str, float] = {
            "attempted": float(len(batch)),
            "collected": float(len(successful)),
            "solved": float(sum(record.outcome == "solved" for record in records)),
            "searches_failed": float(
                sum(record.outcome in {"hard_timeout", "error"} for record in records)
            ),
            "searches_cooperative_deadline": float(
                sum(record.outcome == "cooperative_deadline" for record in records)
            ),
            "searches_hard_timed_out": float(
                sum(record.outcome == "hard_timeout" for record in records)
            ),
            "seeds_inadmissible": float(
                sum(record.outcome == "inadmissible" for record in records)
            ),
            "worker_count": float(len(self.active_workers)),
            "server_count": float(len(self.active_workers)),
            "workers_replaced": float(sum(replacements.values())),
            "worker_replacement_failures": float(sum(replacement_failures.values())),
            "queue_wait_total_s": queue_wait_total,
            "queue_wait_mean_s": queue_wait_total / len(records) if records else 0.0,
            "search_total_s": search_total,
            "search_mean_s": search_total / len(records) if records else 0.0,
            "collection_wall_clock_s": time.perf_counter() - round_started,
            "collection_model_generation": float(model_generation),
        }
        for device in self.plan.collection_devices:
            if device.type == "cuda":
                device_index = device.index or 0
                stats[f"cuda_{device_index}_allocated_bytes"] = float(
                    torch.cuda.memory_allocated(device)
                )
                stats[f"cuda_{device_index}_reserved_bytes"] = float(
                    torch.cuda.memory_reserved(device)
                )
        for worker in workers:
            worker_records = [
                record for record in records if record.worker_id == worker.worker_id
            ]
            stats[f"worker_{worker.worker_id}_attempted"] = float(len(worker_records))
            stats[f"worker_{worker.worker_id}_failed"] = float(
                sum(
                    record.outcome in {"hard_timeout", "error"}
                    for record in worker_records
                )
            )
            stats[f"worker_{worker.worker_id}_hard_timed_out"] = float(
                sum(record.outcome == "hard_timeout" for record in worker_records)
            )
            stats[f"worker_{worker.worker_id}_replaced"] = float(
                replacements[worker.worker_id]
            )
        return CollectionRound(
            results=[record.result for record in successful],
            records=records,
            stats=stats,
        )
