"""Supervised model uses the same source-aware policy as actor-critic collection."""

import torch
from torch import nn
from .policy_model import GraphActionPolicy
from .source_aware_decoder import SourceAwareDecoder
from .policy_actions import ArgumentReference, local_pools
from .model import PublishedGATBackbone


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


# Published v2 checkpoint (Mathlib-Sexpr-GNN) module layout, loaded only by its
# joint bundle loader; a separate format from TacticWithArgsClassifier above.


class PublishedGRUPointer(nn.Module):
    def __init__(self, hidden_dim: int, max_args: int) -> None:
        super().__init__()
        self.init_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.max_args = max_args


class PublishedPointerClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        use_node_type: bool,
        max_args: int,
    ) -> None:
        super().__init__()
        self.backbone = PublishedGATBackbone(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_node_type=use_node_type,
        )
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)
        self.argument_selector = PublishedGRUPointer(hidden_dim, max_args)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.max_args = max_args
        self.hidden_dim = hidden_dim

    def forward(self, data, *, tactic_ids: torch.Tensor | None = None):
        """Run the published flat GRU decoder over one or more PyG graphs.

        ``arg_logits`` are padded per-graph node scores.  The decoder keeps
        its recurrent state per graph and uses the selected node embedding as
        the next input, matching the release's flat argument contract.
        """
        node_embeddings = self.backbone.encode_nodes(data)
        state_embeddings = self.backbone.readout(node_embeddings, data)
        tactic_logits = self.backbone.classifier(state_embeddings)
        batch_size = int(state_embeddings.size(0))
        if tactic_ids is None:
            tactic_ids = tactic_logits.argmax(dim=-1)
        tactic_embeddings = self.tactic_embedding(tactic_ids)
        hidden = torch.tanh(
            self.argument_selector.init_proj(
                torch.cat([state_embeddings, tactic_embeddings], dim=-1)
            )
        )
        batch_index = data.batch
        premise_mask = getattr(data, "premise_mask", None)
        if premise_mask is None:
            premise_mask = torch.ones(
                node_embeddings.size(0), dtype=torch.bool, device=node_embeddings.device
            )
        counts = torch.bincount(batch_index, minlength=batch_size)
        max_nodes = int(counts.max().item())
        padded_nodes = torch.zeros(
            batch_size,
            max_nodes,
            node_embeddings.size(-1),
            device=node_embeddings.device,
            dtype=node_embeddings.dtype,
        )
        padded_mask = torch.zeros(
            batch_size, max_nodes, dtype=torch.bool, device=node_embeddings.device
        )
        offsets = torch.zeros_like(batch_index)
        for graph_id in range(batch_size):
            selected = batch_index == graph_id
            count = int(selected.sum().item())
            offsets[selected] = torch.arange(count, device=batch_index.device)
        padded_nodes[batch_index, offsets] = node_embeddings
        padded_mask[batch_index, offsets] = premise_mask

        arg_logits: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        for _ in range(self.max_args):
            query = self.argument_selector.out_proj(hidden)
            scores = torch.bmm(
                query.unsqueeze(1), padded_nodes.transpose(1, 2)
            ).squeeze(1) / max(self.hidden_dim, 1) ** 0.5
            scores = scores.masked_fill(~padded_mask, float("-inf"))
            arg_logits.append(scores)
            selected = scores.argmax(dim=-1)
            selected_embeddings = padded_nodes[
                torch.arange(batch_size, device=node_embeddings.device), selected
            ]
            stop_logits.append(self.stop_head(hidden).squeeze(-1))
            hidden = self.argument_selector.gru(selected_embeddings, hidden)
        return tactic_logits, arg_logits, torch.stack(stop_logits, dim=1)


class JointSourcePolicy(nn.Module):
    """Exact module layout of the published joint-source v2 checkpoint."""

    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        use_node_type: bool,
        max_args: int,
    ) -> None:
        super().__init__()
        self.pointer = PublishedPointerClassifier(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_node_type=use_node_type,
            max_args=max_args,
        )
        self.arg_count_head = nn.Linear(hidden_dim, max_args + 1)
        # Reference layout: Linear → LayerNorm → Tanh. The Tanh sits at index 2
        # (parameter-less, absent from the checkpoint) and is part of the
        # trained forward computation, not an optional detail.
        self.lemma_adapter = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.library_residual_scale = nn.Parameter(torch.tensor(1.0))
        self.log_inverse_text_temperature = nn.Parameter(torch.tensor(0.0))
        self.source_head = nn.Linear(hidden_dim + 4, 2)

    def forward(self, data, *, tactic_ids: torch.Tensor | None = None):
        tactic_logits, arg_logits, stop_logits = self.pointer(
            data, tactic_ids=tactic_ids
        )
        state_embeddings = self.pointer.backbone.readout(
            self.pointer.backbone.encode_nodes(data), data
        )
        arg_count_logits = self.arg_count_head(state_embeddings)
        return tactic_logits, arg_logits, stop_logits, arg_count_logits, state_embeddings
