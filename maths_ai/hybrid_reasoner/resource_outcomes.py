"""Portable operational outcomes, independent of Lean allocation identities."""

from dataclasses import asdict, dataclass
from enum import Enum

from pantograph_async import PantographResourceError, PantographSessionResourceError


@dataclass(frozen=True)
class ResourceDiagnostic:
    command: str
    kind: str
    phase: str
    limit: int
    recovery: str

    @classmethod
    def from_error(
        cls, error: PantographResourceError | PantographSessionResourceError
    ) -> "ResourceDiagnostic":
        if error.command is None:
            raise ValueError("Resource errors must identify their command.")
        return cls(error.command, error.kind, error.phase, error.limit, error.recovery)

    def as_dict(self) -> dict:
        return asdict(self)


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RESOURCE_LIMIT = "resource_limit"


class ReplayResourceExhausted(RuntimeError):
    """Artifact execution was inconclusive because an operation exhausted Lean."""

    def __init__(self, diagnostic: ResourceDiagnostic):
        self.resource = diagnostic
        super().__init__(f"{diagnostic.command}: {diagnostic.kind} exhausted during {diagnostic.phase}")


@dataclass(frozen=True)
class CommandOutcome:
    status: CommandStatus
    edge_id: int | None = None
    resource: ResourceDiagnostic | None = None

    def __post_init__(self):
        if not isinstance(self.status, CommandStatus):
            raise TypeError("Command outcomes require a typed status.")
        if (self.status is CommandStatus.ACCEPTED) != (self.edge_id is not None):
            raise ValueError("Only accepted commands identify a result edge.")
        if (self.status is CommandStatus.RESOURCE_LIMIT) != (self.resource is not None):
            raise ValueError("Only resource-limited commands carry resource details.")


def resource_from_exception(error: BaseException) -> ResourceDiagnostic | None:
    """Preserve resource context when ownership cleanup wraps an operation."""
    seen = set()
    for _ in range(8):
        if id(error) in seen:
            break
        seen.add(id(error))
        if isinstance(error, (PantographResourceError, PantographSessionResourceError)):
            return ResourceDiagnostic.from_error(error)
        cause = error.__cause__ or error.__context__
        if cause is None:
            break
        error = cause
    return None
