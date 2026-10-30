"""Supervised model uses the same source-aware policy as actor-critic collection."""

import torch
from torch import nn
from .policy_model import GraphActionPolicy
from .source_aware_decoder import SourceAwareDecoder
from .policy_actions import ArgumentReference, local_pools


class TacticWithArgsClassifier(GraphActionPolicy):
    def __init__(
        self,
        *,
        encoder,
        num_tactics,
        dropout=0.2,
        max_args=3,
        text_dim=384,
        local_type_bonus_strength=0.0,
        retrieval_binding="",
    ):
        super().__init__()
        self.retrieval_binding = retrieval_binding
        self.encoder = encoder
        self.hidden_dim, self.max_args = encoder.output_dim, max_args
        self.tactic_classifier = nn.Linear(self.hidden_dim, num_tactics)
        self.tactic_dropout = nn.Dropout(dropout)
        self.decoder = SourceAwareDecoder(
            self.hidden_dim, num_tactics, max_args, text_dim, local_type_bonus_strength
        )

    def encode_graph(self, data):
        self.eval()
        return self.encoder(data)

    def predict_tactics(self, encoded, *, tactic_mask=None):
        logits = self.tactic_classifier(encoded.state_embeddings)
        return (
            logits
            if tactic_mask is None
            else logits.masked_fill(~tactic_mask, -torch.inf)
        )

    def encode(self, data, *, tactic_mask=None):
        encoded = self.encode_graph(data)
        return (
            encoded.node_embeddings,
            encoded.state_embeddings,
            self.predict_tactics(encoded, tactic_mask=tactic_mask),
            encoded.state_embeddings.sum(-1, keepdim=True) * 0,
        )


def supervised_actions(batch, max_args):
    if not hasattr(batch, "arg_count") or not hasattr(batch, "arg_node_indices"):
        raise ValueError(
            "Supervised argument training requires explicit argument targets."
        )
    if sum(batch.arg_count.tolist()) != batch.arg_node_indices.numel():
        raise ValueError("Argument counts and flattened targets differ.")
    rows = []
    offset = 0
    if (
        not hasattr(batch, "arg_lemma_ids")
        or batch.arg_lemma_ids.numel() != batch.arg_node_indices.numel()
    ):
        raise ValueError("Supervised data must align local and library target arrays.")
    lemma_ids = batch.arg_lemma_ids
    for count in batch.arg_count.tolist():
        if not 0 <= count <= max_args:
            raise ValueError("Argument target exceeds the policy budget.")
        args = []
        for local, lemma in zip(
            batch.arg_node_indices[offset : offset + count].tolist(),
            lemma_ids[offset : offset + count].tolist(),
        ):
            if (local >= 0) == (lemma >= 0):
                raise ValueError("Each argument needs exactly one source.")
            args.append(
                ArgumentReference(
                    "local" if local >= 0 else "library",
                    int(local if local >= 0 else lemma),
                )
            )
        rows.append(tuple(args))
        offset += count
    return tuple(rows)


def supervised_loss(
    model, batch, *, candidate_pools, unknown_tactic_id=0, argument_weight=1.0
):
    """Train counts on supported targets and source/identity on reachable prefixes."""
    import torch.nn.functional as F
    from types import SimpleNamespace

    actions = supervised_actions(batch, model.max_args)
    nodes, states, logits, _ = model.encode(batch)
    labels = batch.y.view(-1)
    known = labels != unknown_tactic_id
    tactic_loss = (
        F.cross_entropy(logits[known], labels[known])
        if known.any()
        else logits.sum() * 0
    )
    model._validate_pools(batch, candidate_pools)
    if not hasattr(batch, "action_target_supported"):
        raise ValueError(
            "Supervised data must record action_target_supported; regenerate argument labels."
        )
    supported = batch.action_target_supported.view(-1).bool()
    if supported.numel() != len(actions):
        raise ValueError("Action support metadata must align with graph rows.")
    parts = []
    for row, (pool, gold) in enumerate(zip(candidate_pools, actions)):
        if not known[row] or not supported[row]:
            continue
        decoder = model.decoder
        local = nodes[int(batch.ptr[row]) : int(batch.ptr[row + 1])][
            list(pool.local_ids)
        ]
        adapted = (
            decoder.lemma_adapter(
                pool.library.gather(
                    pool.lemma_ids, device=nodes.device, dtype=nodes.dtype
                )
            )
            if pool.lemma_ids
            else nodes.new_empty((0, model.hidden_dim))
        )
        hidden = decoder.argument_selector.initial_state(
            states[row : row + 1], decoder.tactic_embedding(labels[row : row + 1])
        )
        losses = [
            F.cross_entropy(
                decoder.arg_count_head(hidden), labels.new_tensor([len(gold)])
            )
        ]
        for step, ref in enumerate(gold):
            source, local_logits, library_logits = decoder.score_step(
                hidden,
                local,
                adapted,
                nodes.new_tensor(pool.lemma_scores),
                pool.signal_for(int(labels[row])),
                step,
            )
            target = 0 if ref.source == "local" else 1
            ids, scores, vectors = (
                (pool.local_ids, local_logits, local)
                if target == 0
                else (pool.lemma_ids, library_logits, adapted)
            )
            if not ids:
                break
            losses.append(
                F.cross_entropy(source.view(1, -1), labels.new_tensor([target]))
            )
            if ref.identity not in ids:
                break
            position = ids.index(ref.identity)
            losses.append(
                F.cross_entropy(scores.view(1, -1), labels.new_tensor([position]))
            )
            hidden = decoder.argument_selector.gru(
                vectors[position : position + 1], hidden
            )
        parts.append(torch.stack(losses).sum())
    argument_loss = torch.stack(parts).mean() if parts else logits.sum() * 0
    loss = tactic_loss + argument_weight * argument_loss
    return (
        loss,
        dict(
            tactic_loss=float(tactic_loss.detach()),
            arg_loss=float(argument_loss.detach()),
            total_loss=float(loss.detach()),
        ),
        SimpleNamespace(tactic_logits=logits),
    )
