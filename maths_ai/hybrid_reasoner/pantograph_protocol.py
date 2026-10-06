"""Pure codec for the repository-owned, explicit-focused Pantograph contract.

There is no subprocess or upstream-client dependency here. Missing transition
evidence is rejected, including responses from the unmodified custom REPL.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from maths_ai.data_models.proof_components import Goal, GoalLocal
from maths_ai.gnn_inference.atp_lean_gnn.graph_contract import MODEL_SEXPR_GRAPH_SPEC


PROTOCOL_DESCRIPTION = MappingProxyType(
    {
        "protocol": "maths-ai-pantograph",
        "protocolVersion": 1,
        "modelSexpVersion": MODEL_SEXPR_GRAPH_SPEC.model_sexp_version,
        "transitionValidationVersion": 1,
        "messageVersion": 1,
        "goalSelection": "explicit-focused-v1",
        "tacticErrorsFormat": "string-array-v1",
    }
)
REQUIRED_OPTIONS = MappingProxyType(
    {
        "printJsonPretty": False,
        "printExprPretty": True,
        "printExprAST": True,
        "printExprModelAST": True,
        "noRepeat": False,
        "automaticMode": True,
    }
)
OPTION_NAMES = frozenset(REQUIRED_OPTIONS) | {
    "printDependentMVars",
    "printAuxDecls",
    "printImplementationDetailHyps",
}
BINDER_ROLES = frozenset(
    {
        ":explicit",
        ":implicit",
        ":strict-implicit",
        ":instance-implicit",
        ":let",
    }
)
SUPPORTED_COMMANDS = frozenset(
    {
        "protocol.describe",
        "options.set",
        "options.print",
        "goal.start",
        "goal.tactic",
        "goal.delete",
        "stat",
    }
)
# These categories come from Repl.lean and Library.lean at the pinned base.
# "validation" is reserved for the required companion extension.
COMMAND_ERROR_CATEGORIES = MappingProxyType(
    {
        "protocol.describe": frozenset({"command"}),
        "options.set": frozenset({"command"}),
        "options.print": frozenset({"command"}),
        "goal.start": frozenset({"command", "arguments", "parsing", "elab"}),
        "goal.tactic": frozenset(
            {"command", "arguments", "index", "invalid", "validation"}
        ),
        "goal.delete": frozenset({"command"}),
        "stat": frozenset({"command"}),
    }
)


class PantographError(RuntimeError):
    """Base failure with command and independently copied wire-payload context."""

    def __init__(
        self, message: str, *, command: str | None = None, payload: Any = None
    ):
        super().__init__(message)
        self.command = command
        self._payload = deepcopy(payload)

    @property
    def payload(self) -> Any:
        return deepcopy(self._payload)


class PantographConfigurationError(PantographError):
    pass


class PantographStartupError(PantographError):
    pass


class PantographGoalRejected(PantographError):
    pass


class PantographTacticRejected(PantographError):
    pass


class PantographProtocolError(PantographError):
    pass


class PantographTransportError(PantographError):
    pass


class PantographRequestTimeout(PantographTransportError):
    pass


class PantographStateError(PantographError):
    pass


@dataclass(frozen=True)
class StateHandle:
    session_id: str
    scope_id: str
    state_id: int

    def __post_init__(self):
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or not isinstance(self.scope_id, str)
            or not self.scope_id
            or type(self.state_id) is not int
            or self.state_id < 0
        ):
            raise PantographStateError(
                "A handle requires session/scope identities and a nonnegative integer ID."
            )


@dataclass(frozen=True)
class WireExpression:
    pp: str
    sexp: str
    model_sexp: str
    model_sexp_version: int
    dependent_mvars: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExecutionLocal:
    internal_name: str
    user_name: str
    context_index: int
    binder_role: str
    is_instance: bool
    is_let: bool
    is_inaccessible: bool
    type: WireExpression
    value: WireExpression | None = None

    def to_goal_local(self) -> GoalLocal:
        return GoalLocal(
            user_name=self.user_name or "_",
            internal_name=self.internal_name,
            context_index=self.context_index,
            binder_role=self.binder_role,
            is_instance=self.is_instance,
            is_let=self.is_let,
            type_pp=self.type.pp,
            type_model_sexp=self.type.model_sexp,
            value_pp=self.value.pp if self.value else None,
            value_model_sexp=self.value.model_sexp if self.value else None,
            model_sexp_version=self.type.model_sexp_version,
        )


@dataclass(frozen=True)
class ExecutionGoal:
    goal_name: str
    case_tag: str | None
    is_conversion: bool
    target: WireExpression
    locals: tuple[ExecutionLocal, ...]

    @property
    def goal(self) -> Goal:
        """Return a fresh canonical copy; mutations cannot alter this snapshot."""
        return execution_goal_to_goal(self)


@dataclass(frozen=True)
class MessagePosition:
    line: int
    column: int


@dataclass(frozen=True)
class ExecutionMessage:
    severity: str
    data: str
    pos: MessagePosition | None = None
    end_pos: MessagePosition | None = None
    file_name: str | None = None
    kind: str | None = None


@dataclass(frozen=True)
class TransitionValidation:
    version: int
    scope: str
    checked: bool
    has_sorry: bool
    has_unsafe: bool

    @property
    def is_rejection(self) -> bool:
        return self.has_sorry or self.has_unsafe


@dataclass(frozen=True)
class StartedGoal:
    handle: StateHandle
    root: str


@dataclass(frozen=True)
class TacticState:
    handle: StateHandle
    goals: tuple[ExecutionGoal, ...]
    validation: TransitionValidation
    messages: tuple[ExecutionMessage, ...]


@dataclass(frozen=True)
class GoalStartResult:
    state_id: int
    root: str


@dataclass(frozen=True)
class TacticSuccess:
    next_state_id: int
    goals: tuple[ExecutionGoal, ...]
    validation: TransitionValidation
    messages: tuple[ExecutionMessage, ...]


@dataclass(frozen=True)
class TacticParseFailure:
    parse_error: str
    messages: tuple[ExecutionMessage, ...]


@dataclass(frozen=True)
class TacticFailure:
    tactic_errors: tuple[str, ...]
    messages: tuple[ExecutionMessage, ...]


@dataclass(frozen=True)
class CommandError:
    command: str
    category: str
    description: str

    def as_exception(self) -> PantographError:
        error_class: type[PantographError] = PantographProtocolError
        if self.command == "goal.start" and self.category in {"parsing", "elab"}:
            error_class = PantographGoalRejected
        elif self.command == "goal.tactic" and self.category == "index":
            error_class = PantographStateError
        return error_class(
            self.description,
            command=self.command,
            payload={"error": self.category, "desc": self.description},
        )


@dataclass(frozen=True)
class PantographStats:
    n_goals: int


@dataclass(frozen=True)
class CommandAcknowledgement:
    command: str


@dataclass(frozen=True)
class ProtocolDescription:
    protocol: str
    protocol_version: int
    model_sexp_version: int
    transition_validation_version: int
    message_version: int
    goal_selection: str
    tactic_errors_format: str


@dataclass(frozen=True)
class SessionOptions:
    values: tuple[tuple[str, bool], ...]

    def to_dict(self) -> dict[str, bool]:
        return dict(self.values)


Response = (
    GoalStartResult
    | TacticSuccess
    | TacticParseFailure
    | TacticFailure
    | CommandError
    | PantographStats
    | CommandAcknowledgement
    | ProtocolDescription
    | SessionOptions
)


def _object(value: Any, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise PantographProtocolError(f"{owner} must be a JSON object.")
    return value


def _fields(
    value: dict[str, Any], required: set[str] | frozenset[str], optional=()
) -> None:
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise PantographProtocolError(
            f"Invalid fields: missing={sorted(missing)}, unexpected={sorted(extra)}."
        )


def _string(value: Any, owner: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise PantographProtocolError(
            f"{owner} must be a{' nonempty' if nonempty else ''} string."
        )
    # Reject unpaired surrogates even when JSON used escaped character values.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PantographProtocolError(f"{owner} is not valid UTF-8 text.") from exc
    return value


def _nat(value: Any, owner: str) -> int:
    if type(value) is not int or value < 0:
        raise PantographProtocolError(
            f"{owner} must be a nonnegative integer, not a Boolean."
        )
    return value


def _bool(value: Any, owner: str) -> bool:
    if type(value) is not bool:
        raise PantographProtocolError(f"{owner} must be a Boolean.")
    return value


def _array(value: Any, owner: str) -> list[Any]:
    if not isinstance(value, list):
        raise PantographProtocolError(f"{owner} must be a JSON array.")
    return value


def decode_json_response(line: str | bytes) -> dict[str, Any]:
    """Parse one response object without accepting duplicate keys or NaN."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PantographProtocolError(f"Duplicate JSON key {key!r}.")
            result[key] = value
        return result

    def constant(value):
        raise PantographProtocolError(f"Nonstandard JSON constant {value!r}.")

    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="strict")
        if not isinstance(line, str):
            raise PantographProtocolError(
                "A JSON response requires text or UTF-8 bytes."
            )
        return _object(
            json.loads(line, object_pairs_hook=pairs, parse_constant=constant),
            "response",
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise PantographProtocolError(f"Invalid UTF-8/JSON response: {exc}.") from exc


def encode_request(command: str, payload: Mapping[str, Any]) -> bytes:
    """Validate the supported request and produce exactly one compact line."""
    if command not in SUPPORTED_COMMANDS:
        raise PantographConfigurationError(
            f"Unsupported command {command!r}.", command=command
        )
    body = _object(dict(payload), "request")
    if command in {"protocol.describe", "options.print", "stat"}:
        _fields(body, set())
    elif command == "options.set":
        _fields(body, set(), OPTION_NAMES)
        for key, value in body.items():
            _bool(value, key)
            if key in REQUIRED_OPTIONS and value is not REQUIRED_OPTIONS[key]:
                raise PantographConfigurationError(
                    f"Option {key} conflicts with the execution contract."
                )
    elif command == "goal.start":
        _fields(body, {"expr"})
        _string(body["expr"], "expr", nonempty=True)
    elif command == "goal.tactic":
        _fields(body, {"stateId", "goalId", "tactic"})
        _nat(body["stateId"], "stateId")
        _nat(body["goalId"], "goalId")
        _string(body["tactic"], "tactic", nonempty=True)
    else:
        _fields(body, {"stateIds"})
        ids = [
            _nat(item, "stateIds entry")
            for item in _array(body["stateIds"], "stateIds")
        ]
        if len(set(ids)) != len(ids):
            raise PantographStateError("Deletion IDs must be unique.")
    return (
        command
        + " "
        + json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def decode_expression(payload: Any) -> WireExpression:
    obj = _object(payload, "expression")
    _fields(obj, {"pp", "sexp", "modelSexp", "modelSexpVersion"}, {"dependentMVars"})
    version = _nat(obj["modelSexpVersion"], "modelSexpVersion")
    if version != MODEL_SEXPR_GRAPH_SPEC.model_sexp_version:
        raise PantographProtocolError(f"Unsupported modelSexpVersion {version}.")
    deps = None
    if "dependentMVars" in obj:
        deps = tuple(
            _string(item, "dependent metavariable", nonempty=True)
            for item in _array(obj["dependentMVars"], "dependentMVars")
        )
    return WireExpression(
        _string(obj["pp"], "pp", nonempty=True),
        _string(obj["sexp"], "sexp", nonempty=True),
        _string(obj["modelSexp"], "modelSexp", nonempty=True),
        version,
        deps,
    )


def decode_execution_goal(payload: Any) -> ExecutionGoal:
    obj = _object(payload, "goal")
    _fields(obj, {"name", "isConversion", "target", "vars"}, {"userName"})
    target = decode_expression(obj["target"])
    locals_ = []
    for item in _array(obj["vars"], "vars"):
        local = _object(item, "local")
        _fields(
            local,
            {
                "name",
                "userName",
                "contextIndex",
                "binderRole",
                "isInstance",
                "isLet",
                "isInaccessible",
                "type",
            },
            {"value"},
        )
        role = _string(local["binderRole"], "binderRole")
        if role not in BINDER_ROLES:
            raise PantographProtocolError(f"Unsupported binderRole {role!r}.")
        instance = _bool(local["isInstance"], "isInstance")
        is_let = _bool(local["isLet"], "isLet")
        if is_let != (role == ":let") or instance != (role == ":instance-implicit"):
            raise PantographProtocolError(
                "Local binder role and instance/let flags disagree."
            )
        if ("value" in local) != is_let:
            raise PantographProtocolError(
                "Only let locals must carry a complete value expression."
            )
        locals_.append(
            ExecutionLocal(
                _string(local["name"], "local name", nonempty=True),
                _string(local["userName"], "local userName"),
                _nat(local["contextIndex"], "contextIndex"),
                role,
                instance,
                is_let,
                _bool(local["isInaccessible"], "isInaccessible"),
                decode_expression(local["type"]),
                decode_expression(local["value"]) if is_let else None,
            )
        )
    indices = [local.context_index for local in locals_]
    if indices != sorted(set(indices)):
        raise PantographProtocolError(
            "Locals must have unique increasing context indices; gaps are permitted."
        )
    return ExecutionGoal(
        _string(obj["name"], "goal name", nonempty=True),
        _string(obj["userName"], "goal userName") if "userName" in obj else None,
        _bool(obj["isConversion"], "isConversion"),
        target,
        tuple(locals_),
    )


def execution_goal_to_goal(goal: ExecutionGoal) -> Goal:
    return Goal(
        expression=goal.target.pp,
        goal_model_sexp=goal.target.model_sexp,
        locals=[local.to_goal_local() for local in goal.locals],
        case_tag=goal.case_tag,
        model_sexp_version=goal.target.model_sexp_version,
    )


def execution_state_to_goals(state: TacticState | TacticSuccess) -> list[Goal]:
    return [execution_goal_to_goal(goal) for goal in state.goals]


def execution_goal_to_translator(goal: ExecutionGoal) -> dict[str, Any]:
    """Own named projection; translator consumers migrate to this in Stage 6."""

    def expression(value: WireExpression) -> dict[str, Any]:
        return {
            "pp": value.pp,
            "sexp": value.sexp,
            "model_sexp": value.model_sexp,
            "model_sexp_version": value.model_sexp_version,
        }

    return {
        "goal_name": goal.goal_name,
        "case_tag": goal.case_tag,
        "is_conversion": goal.is_conversion,
        "target": expression(goal.target),
        "locals": [
            {
                "internal_name": local.internal_name,
                "user_name": local.user_name,
                "context_index": local.context_index,
                "binder_role": local.binder_role,
                "is_instance": local.is_instance,
                "is_let": local.is_let,
                "is_inaccessible": local.is_inaccessible,
                "type": expression(local.type),
                "value": expression(local.value) if local.value is not None else None,
            }
            for local in goal.locals
        ],
    }


def decode_messages(payload: Any) -> tuple[ExecutionMessage, ...]:
    def position(value: Any) -> MessagePosition:
        obj = _object(value, "message position")
        _fields(obj, {"line", "column"})
        line = _nat(obj["line"], "line")
        if line == 0:
            raise PantographProtocolError("Message lines are one-based.")
        return MessagePosition(line, _nat(obj["column"], "column"))

    result = []
    for item in _array(payload, "messages"):
        obj = _object(item, "message")
        _fields(obj, {"severity", "data"}, {"pos", "endPos", "fileName", "kind"})
        severity = _string(obj["severity"], "severity")
        if severity not in {"information", "warning", "error"}:
            raise PantographProtocolError(f"Unknown message severity {severity!r}.")
        pos = position(obj["pos"]) if "pos" in obj else None
        end = position(obj["endPos"]) if "endPos" in obj else None
        if end is not None and (
            pos is None or (end.line, end.column) < (pos.line, pos.column)
        ):
            raise PantographProtocolError(
                "Message endPos requires an ordered start position."
            )
        result.append(
            ExecutionMessage(
                severity,
                _string(obj["data"], "message data"),
                pos,
                end,
                _string(obj["fileName"], "fileName") if "fileName" in obj else None,
                _string(obj["kind"], "kind") if "kind" in obj else None,
            )
        )
    return tuple(result)


def decode_transition_validation(payload: Any) -> TransitionValidation:
    obj = _object(payload, "validation")
    _fields(obj, {"version", "scope", "checked", "hasSorry", "hasUnsafe"})
    version = _nat(obj["version"], "validation version")
    scope = _string(obj["scope"], "validation scope")
    checked = _bool(obj["checked"], "validation checked")
    if version != 1 or scope != "transition-expressions" or not checked:
        raise PantographProtocolError(
            "Required transition validation was not performed with the supported version/scope."
        )
    return TransitionValidation(
        version,
        scope,
        checked,
        _bool(obj["hasSorry"], "hasSorry"),
        _bool(obj["hasUnsafe"], "hasUnsafe"),
    )


def decode_response(command: str, payload: Any) -> Response:
    """Decode one final-contract response, retaining typed domain failures.

    Positive admission/unsafe flags remain a TacticSuccess rejection candidate.
    The future state owner must register and delete that allocation before raising.
    """
    if command not in SUPPORTED_COMMANDS:
        raise PantographConfigurationError(
            f"Unsupported command {command!r}.", command=command
        )
    try:
        obj = _object(payload, "response")
        if "error" in obj:
            _fields(obj, {"error", "desc"})
            category = _string(obj["error"], "error category")
            if category not in COMMAND_ERROR_CATEGORIES[command]:
                raise PantographProtocolError(
                    f"Unknown {command} error category {category!r}."
                )
            return CommandError(
                command, category, _string(obj["desc"], "error description")
            )
        if command == "goal.start":
            _fields(obj, {"stateId", "root"})
            return GoalStartResult(
                _nat(obj["stateId"], "stateId"),
                _string(obj["root"], "root", nonempty=True),
            )
        if command == "goal.tactic":
            variants = [
                key
                for key in ("nextStateId", "parseError", "tacticErrors")
                if key in obj
            ]
            if len(variants) != 1:
                raise PantographProtocolError(
                    "A tactic response must contain exactly one success/failure indicator."
                )
            variant = variants[0]
            if variant == "nextStateId":
                _fields(obj, {"nextStateId", "goals", "validation", "messages"})
                goals = tuple(
                    decode_execution_goal(item)
                    for item in _array(obj["goals"], "goals")
                )
                if len({goal.goal_name for goal in goals}) != len(goals):
                    raise PantographProtocolError(
                        "Execution goal names must be unique within a state."
                    )
                messages = decode_messages(obj["messages"])
                if any(message.severity == "error" for message in messages):
                    raise PantographProtocolError(
                        "A logged error cannot accompany a successful transition."
                    )
                return TacticSuccess(
                    _nat(obj["nextStateId"], "nextStateId"),
                    goals,
                    decode_transition_validation(obj["validation"]),
                    messages,
                )
            _fields(obj, {variant, "messages"})
            messages = decode_messages(obj["messages"])
            if variant == "parseError":
                return TacticParseFailure(
                    _string(obj[variant], variant, nonempty=True), messages
                )
            errors = tuple(
                _string(item, "tactic error", nonempty=True)
                for item in _array(obj[variant], variant)
            )
            if not errors:
                raise PantographProtocolError(
                    "tacticErrors must contain at least one error."
                )
            return TacticFailure(errors, messages)
        if command == "stat":
            _fields(obj, {"nGoals"})
            return PantographStats(_nat(obj["nGoals"], "nGoals"))
        if command in {"goal.delete", "options.set"}:
            _fields(obj, set())
            return CommandAcknowledgement(command)
        if command == "protocol.describe":
            _fields(obj, frozenset(PROTOCOL_DESCRIPTION))
            for key, expected in PROTOCOL_DESCRIPTION.items():
                value = (
                    _nat(obj[key], key)
                    if type(expected) is int
                    else _string(obj[key], key)
                )
                if value != expected:
                    raise PantographProtocolError(
                        f"Unsupported protocol descriptor {key}={value!r}; expected {expected!r}."
                    )
            return ProtocolDescription(
                obj["protocol"],
                obj["protocolVersion"],
                obj["modelSexpVersion"],
                obj["transitionValidationVersion"],
                obj["messageVersion"],
                obj["goalSelection"],
                obj["tacticErrorsFormat"],
            )
        _fields(obj, OPTION_NAMES)
        for key in OPTION_NAMES:
            _bool(obj[key], key)
            if key in REQUIRED_OPTIONS and obj[key] is not REQUIRED_OPTIONS[key]:
                raise PantographProtocolError(
                    f"Effective option {key} conflicts with the execution contract."
                )
        return SessionOptions(tuple(sorted(obj.items())))
    except PantographProtocolError as exc:
        # Decoder helpers know the field; the envelope supplies command context.
        raise PantographProtocolError(
            str(exc), command=command, payload=payload
        ) from exc
