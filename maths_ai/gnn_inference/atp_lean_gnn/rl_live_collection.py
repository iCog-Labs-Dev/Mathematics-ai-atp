"""Lifecycle and bounded concurrency for live RL theorem collection."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

import torch

from .rl_resources import ResourcePlan


CollectionOutcome = Literal[
    "solved",
    "unsolved",
    "cooperative_deadline",
    "hard_timeout",
    "error",
]
ServerFactory = Callable[[int], Awaitable[Any]]
ReasonerFactory = Callable[[int, torch.device, torch.nn.Module, Any], Any]
SearchFactory = Callable[[Any, Any, bool, float], Awaitable[Any]]
SolvedPredicate = Callable[[Any], bool]
CooperativeDeadlinePredicate = Callable[[Any], bool]
LogFunction = Callable[[str], None]


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


def _close_server(server: Any) -> None:
    try:
        server._close()
    except Exception:
        return


async def _close_servers(servers: list[Any], timeout_s: float) -> None:
    await asyncio.wait_for(
        asyncio.gather(
            *(asyncio.to_thread(_close_server, server) for server in servers)
        ),
        timeout=timeout_s,
    )


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
                server = await asyncio.wait_for(
                    server_factory(worker_id), timeout=startup_timeout_s
                )
                reasoner = reasoner_factory(worker_id, device, replica, server)
                return LiveReasonerWorker(worker_id, device, replica, reasoner, server)
            except BaseException as exc:
                if server is not None:
                    _close_server(server)
                raise WorkerStartupError(
                    worker_id=worker_id,
                    device=device,
                    environment_description=environment_description,
                    cause=exc,
                ) from exc

        results = await asyncio.gather(
            *(create_one(*binding) for binding in bindings), return_exceptions=True
        )
        workers = [result for result in results if isinstance(result, LiveReasonerWorker)]
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            await _close_servers(
                [worker.server for worker in workers], shutdown_timeout_s
            )
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
        try:
            await self.close()
        except asyncio.TimeoutError:
            self.log("Warning: timed out while closing Pantograph worker servers.")

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
        """Close every current server once, bounded by the configured timeout."""

        if self._closed:
            return
        self._closed = True
        await _close_servers(
            [worker.server for worker in self.workers], self.shutdown_timeout_s
        )

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
        await _close_servers([worker.server], self.shutdown_timeout_s)
        server = None
        try:
            server = await asyncio.wait_for(
                self.server_factory(worker.worker_id), timeout=self.startup_timeout_s
            )
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
                _close_server(server)
            worker.active = False
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
        greedy: bool = False,
    ) -> CollectionRound:
        """Collect one bounded round and return records in input order."""

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
                try:
                    cooperative_deadline = time.monotonic() + timeout_s
                    result = await asyncio.wait_for(
                        self.search_factory(
                            worker.reasoner,
                            theorem,
                            greedy,
                            cooperative_deadline,
                        ),
                        timeout=timeout_s * 1.25,
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
                except asyncio.TimeoutError as exc:
                    records.append(
                        CollectionRecord(
                            batch_index=index,
                            worker_id=worker.worker_id,
                            queue_wait_s=queue_wait,
                            search_s=time.perf_counter() - search_started,
                            outcome="hard_timeout",
                            exception_category=type(exc).__name__,
                        )
                    )
                    self.log(
                        f"  [collect worker={worker.worker_id}] hard timeout; "
                        "replacing Pantograph worker"
                    )
                    try:
                        worker = await self._replacement_worker(worker)
                        replacements[worker.worker_id] += 1
                    except WorkerStartupError as replacement_error:
                        replacement_failures[worker.worker_id] += 1
                        active_count -= 1
                        self.log(f"  [collect worker={worker.worker_id}] {replacement_error}")
                        if active_count == 0:
                            raise RuntimeError(
                                "Every live collection worker became unavailable."
                            ) from replacement_error
                        return
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
        except BaseException:
            producer.cancel()
            for task in tasks:
                task.cancel()
            await asyncio.gather(producer, *tasks, return_exceptions=True)
            raise

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
