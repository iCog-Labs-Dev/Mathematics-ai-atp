from __future__ import annotations

from dataclasses import dataclass

from .dataset import DatasetRow
from .graph import DAGBuilder, proof_state_to_dag
from .labels import label_example
from .state import ProofState, parse_state


PREPARATION_PHASES = ("parse_state", "proof_state_to_dag", "label_example")


@dataclass(frozen=True)
class PreparedExample:
    row: DatasetRow
    parsed_state: ProofState
    dag: DAGBuilder
    tactic_name: str


class PreparationPhaseError(Exception):
    def __init__(self, *, phase: str, cause: Exception):
        self.phase = phase
        self.cause = cause
        super().__init__(str(cause))


def prepare_example(row: DatasetRow) -> PreparedExample:
    try:
        parsed_state = parse_state(row.state)
    except Exception as exc:  # pragma: no cover - exercised via callers/tests
        raise PreparationPhaseError(phase="parse_state", cause=exc) from exc

    try:
        dag = proof_state_to_dag(parsed_state)
    except Exception as exc:  # pragma: no cover - exercised via callers/tests
        raise PreparationPhaseError(phase="proof_state_to_dag", cause=exc) from exc

    try:
        label_info = label_example(row.tactic)
    except Exception as exc:  # pragma: no cover - exercised via callers/tests
        raise PreparationPhaseError(phase="label_example", cause=exc) from exc

    return PreparedExample(
        row=row,
        parsed_state=parsed_state,
        dag=dag,
        tactic_name=str(label_info["tactic_name"]),
    )
