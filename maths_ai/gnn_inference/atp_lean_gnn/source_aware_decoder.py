"""Count/source/identity recurrent policy; independent of encoder and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import nn
from .policy_actions import ArgumentReference, CandidatePoolSnapshot


def _choose(dist, logits, *, greedy, generator):
    """The argmax, or a draw from ``dist`` (with ``generator`` when given)."""
    if greedy:
        return logits.argmax()
    if generator is None:
        return dist.sample()
    return torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)


class RecurrentPointer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.init_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self._scale = 1 / math.sqrt(hidden_dim)

    def initial_state(self, state, tactic):
        return torch.tanh(self.init_proj(torch.cat((state, tactic), dim=-1)))

    def score_candidates(self, hidden, vectors):
        return (vectors @ self.out_proj(hidden).unsqueeze(-1)).squeeze(-1) * self._scale


@dataclass
class DecoderEvaluation:
    arguments: tuple[ArgumentReference, ...]
    count_logp: torch.Tensor
    source_logp: torch.Tensor
    identity_logp: torch.Tensor

    @property
    def logp(self):
        return self.count_logp + self.source_logp + self.identity_logp


class SourceAwareDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_tactics,
        max_args=3,
        text_dim=384,
        local_type_bonus_strength=0.0,
    ):
        super().__init__()
        if min(hidden_dim, num_tactics, text_dim, max_args) < 1:
            raise ValueError("Decoder dimensions must be positive.")
        if (
            not math.isfinite(local_type_bonus_strength)
            or local_type_bonus_strength < 0
        ):
            raise ValueError("Type bonus must be finite and nonnegative.")
        self.hidden_dim, self.max_args, self.text_dim = hidden_dim, max_args, text_dim
        self.local_type_bonus_strength = local_type_bonus_strength
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)
        self.argument_selector = RecurrentPointer(hidden_dim)
        self.lemma_adapter = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.source_head = nn.Linear(hidden_dim + 4, 2)
        self.arg_count_head = nn.Linear(hidden_dim, max_args + 1)
        self.log_inverse_text_temperature = nn.Parameter(torch.tensor(3.0))
        self.library_residual_scale = nn.Parameter(torch.zeros(()))

    def score_step(
        self, hidden, local, library_vectors, retrieval_scores, signal, step
    ):
        local_logits = self.argument_selector.score_candidates(hidden, local).flatten()
        library_logits = (
            retrieval_scores * self.log_inverse_text_temperature.clamp(-4, 5).exp()
        )
        if len(library_vectors):
            library_logits = (
                library_logits
                + self.library_residual_scale.tanh()
                * self.argument_selector.score_candidates(
                    hidden, library_vectors
                ).flatten()
            )
        evidence = hidden.new_tensor(
            [
                float(local_logits.detach().max()) if local_logits.numel() else 0.0,
                float(library_logits.detach().max()) if library_logits.numel() else 0.0,
                len(local),
                len(library_vectors),
            ]
        )
        source_logits = self.source_head(torch.cat((hidden.flatten(), evidence)))
        source_logits = source_logits.masked_fill(
            hidden.new_tensor(
                [not len(local), not len(library_vectors)], dtype=torch.bool
            ),
            -torch.inf,
        )
        if step == 0 and signal and self.local_type_bonus_strength:
            local_logits = (
                local_logits
                + self.local_type_bonus_strength * hidden.new_tensor(signal)
            )
        return source_logits, local_logits, library_logits

    def run(self, state, nodes, tactic_id, pool, *, greedy=False, forced=None, generator=None):
        return self.run_many(
            [(state, nodes, tactic_id, pool)],
            greedy=greedy,
            forced_rows=None if forced is None else [forced],
            generator=generator,
        )[0]

    def run_many(self, rows, *, greedy=False, forced_rows=None, generator=None):
        """Share trainable library adaptation within this forward only."""
        cache = {}
        results = []
        if forced_rows is not None and len(forced_rows) != len(rows):
            raise ValueError("Forced arguments must align with decoder rows.")
        for row, (state, nodes, tactic_id, pool) in enumerate(rows):
            key = (pool.artifact_fingerprint, pool.lemma_ids, nodes.device, nodes.dtype)
            if key not in cache:
                if pool.lemma_ids:
                    vectors = pool.library.gather(
                        pool.lemma_ids, device=nodes.device, dtype=nodes.dtype
                    )
                    if vectors.shape[1] != self.text_dim:
                        raise ValueError(
                            "Text embedding dimension disagrees with the policy."
                        )
                    cache[key] = self.lemma_adapter(vectors)
                else:
                    cache[key] = nodes.new_empty((0, self.hidden_dim))
            results.append(
                self._run(
                    state,
                    nodes,
                    tactic_id,
                    pool,
                    adapted=cache[key],
                    greedy=greedy,
                    forced=None if forced_rows is None else forced_rows[row],
                    generator=generator,
                )
            )
        return results

    def _run(self, state, nodes, tactic_id, pool, *, adapted, greedy, forced, generator=None):
        if forced is not None:
            pool.validate(forced)
            if len(forced) > self.max_args:
                raise ValueError("Stored argument count exceeds the policy budget.")
        local = nodes[list(pool.local_ids)]
        hidden = self.argument_selector.initial_state(
            state.view(1, -1), self.tactic_embedding(tactic_id.reshape(1))
        )
        logits = self.arg_count_head(hidden).flatten()
        if not pool.local_ids and not pool.lemma_ids:
            logits = logits.masked_fill(
                torch.arange(self.max_args + 1, device=nodes.device) > 0, -torch.inf
            )
        dist = torch.distributions.Categorical(logits=logits)
        count = (
            torch.tensor(len(forced), device=nodes.device)
            if forced is not None
            else _choose(dist, logits, greedy=greedy, generator=generator)
        )
        count_logp = dist.log_prob(count)
        source_logp = state.sum() * 0
        identity_logp = state.sum() * 0
        args = []
        for step in range(int(count)):
            sources, locals_, lemmas = self.score_step(
                hidden,
                local,
                adapted,
                nodes.new_tensor(pool.lemma_scores),
                pool.signal_for(int(tactic_id)),
                step,
            )
            source_dist = torch.distributions.Categorical(logits=sources)
            source = (
                torch.tensor(
                    0 if forced[step].source == "local" else 1, device=nodes.device
                )
                if forced is not None
                else _choose(source_dist, sources, greedy=greedy, generator=generator)
            )
            ids, scores, values = (
                (pool.local_ids, locals_, local)
                if int(source) == 0
                else (pool.lemma_ids, lemmas, adapted)
            )
            identity_dist = torch.distributions.Categorical(logits=scores)
            position = (
                torch.tensor(ids.index(forced[step].identity), device=nodes.device)
                if forced is not None
                else _choose(identity_dist, scores, greedy=greedy, generator=generator)
            )
            source_logp = source_logp + source_dist.log_prob(source)
            identity_logp = identity_logp + identity_dist.log_prob(position)
            args.append(
                ArgumentReference(
                    "local" if int(source) == 0 else "library", ids[int(position)]
                )
            )
            hidden = self.argument_selector.gru(values[position].reshape(1, -1), hidden)
        return DecoderEvaluation(tuple(args), count_logp, source_logp, identity_logp)
