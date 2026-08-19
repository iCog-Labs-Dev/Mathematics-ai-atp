from __future__ import annotations

from .architectures import build_encoder
from .model import SupervisedTacticClassifier
from .model_spec import ModelSpec
from .pyg import NODE_TYPE_TO_ID


def _build_encoder(
    *,
    model_spec: ModelSpec,
    num_node_labels: int,
):
    return build_encoder(
        architecture=model_spec.architecture,
        encoder_config=model_spec.encoder,
        num_node_labels=num_node_labels,
        num_node_types=len(NODE_TYPE_TO_ID),
        hidden_dim=model_spec.hidden_dim,
        dropout=model_spec.dropout,
        use_node_type=model_spec.use_node_type,
    )


def build_supervised_tactic_model(
    *,
    model_spec: ModelSpec,
    num_node_labels: int,
    num_tactics: int,
) -> SupervisedTacticClassifier:
    model = SupervisedTacticClassifier(
        encoder=_build_encoder(
            model_spec=model_spec,
            num_node_labels=num_node_labels,
        ),
        num_tactics=num_tactics,
        dropout=model_spec.dropout,
    )
    model.model_spec = model_spec
    model.model_kind = "supervised_tactic"
    return model


def build_pointer_model(
    *,
    model_spec: ModelSpec,
    num_node_labels: int,
    num_tactics: int,
):
    from .argument_selector import TacticWithArgsClassifier

    model = TacticWithArgsClassifier(
        encoder=_build_encoder(
            model_spec=model_spec,
            num_node_labels=num_node_labels,
        ),
        num_tactics=num_tactics,
        dropout=model_spec.dropout,
        max_args=model_spec.max_args,
    )
    model.model_spec = model_spec
    model.model_kind = "tactic_with_args"
    return model


def build_actor_critic_model(
    *,
    model_spec: ModelSpec,
    num_node_labels: int,
    num_tactics: int,
):
    from .actor_critic import ActorCriticWithArgsClassifier

    model = ActorCriticWithArgsClassifier(
        encoder=_build_encoder(
            model_spec=model_spec,
            num_node_labels=num_node_labels,
        ),
        num_tactics=num_tactics,
        dropout=model_spec.dropout,
        max_args=model_spec.max_args,
    )
    model.model_spec = model_spec
    model.model_kind = "actor_critic_with_args"
    return model
