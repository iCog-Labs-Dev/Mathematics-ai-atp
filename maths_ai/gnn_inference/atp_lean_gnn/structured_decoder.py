"""Trainable typed-token GRU with input, library and lexical pointer heads.

Graph/text encoders are external frozen feature producers. Validation here is
teacher-forced with gold tactic conditioning, not autonomous proof success.
"""

from collections import Counter
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


FORMAT = "structured_source_decoder_v1"


class StructuredDecoder(nn.Module):
    def __init__(self, frozen_policy, vocabulary):
        super().__init__()
        self.hidden_dim = frozen_policy.pointer.hidden_dim
        self.text_dim = frozen_policy.lemma_adapter[0].in_features
        self.tokens = vocabulary.tokens
        self.token_embedding = nn.Embedding(len(self.tokens), self.hidden_dim, padding_idx=0)
        self.token_head = nn.Linear(self.hidden_dim, len(self.tokens))
        self.tactic_embedding = copy.deepcopy(frozen_policy.pointer.tactic_embedding)
        self.initial_projection = copy.deepcopy(frozen_policy.pointer.argument_selector.init_proj)
        self.gru = copy.deepcopy(frozen_policy.pointer.argument_selector.gru)
        self.pointer_projection = copy.deepcopy(frozen_policy.pointer.argument_selector.out_proj)
        self.lemma_adapter = copy.deepcopy(frozen_policy.lemma_adapter)
        self.log_inverse_text_temperature = nn.Parameter(frozen_policy.log_inverse_text_temperature.detach().clone())
        self.library_residual_scale = nn.Parameter(frozen_policy.library_residual_scale.detach().clone())
        self.lexical_key = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.Tanh())
        self.requires_grad_(True)

    def configuration(self):
        return {"model_type": FORMAT, "hidden_dim": self.hidden_dim, "text_dim": self.text_dim,
                "vocabulary_size": len(self.tokens), "tactic_count": self.tactic_embedding.num_embeddings,
                "tactic_conditioning": "gold_tactic_id_teacher_forcing_v1"}

    def initial_state(self, states, tactic_ids):
        return torch.tanh(self.initial_projection(torch.cat([states, self.tactic_embedding(tactic_ids)], dim=-1)))

    def advance(self, previous_tokens, previous_reference_vectors, hidden):
        return self.gru(self.token_embedding(previous_tokens) + previous_reference_vectors, hidden)

    def pointer_scores(self, hidden, keys):
        return keys @ self.pointer_projection(hidden) / math.sqrt(self.hidden_dim)

    def forward(self, rows):
        """Return row-balanced syntax loss plus reachable pointer CE and counts.

        Unretrieved gold never enters candidate keys. Its matched frozen vector
        may feed the *next* teacher-forced step; unreachable actions count as
        misses, including during teacher-forced validation.
        """
        if not rows:
            raise ValueError("nonempty structured batch required")
        device = next(self.parameters()).device
        count = len(rows)
        lengths = [len(row["token_ids"]) - 1 for row in rows]
        if min(lengths) < 1:
            raise ValueError("nonempty compiled action required")
        tokens = torch.zeros(count, max(lengths) + 1, device=device, dtype=torch.long)
        for i, row in enumerate(rows):
            tokens[i, :len(row["token_ids"])] = torch.tensor(row["token_ids"], device=device)
        states = torch.stack([row["state"] for row in rows]).to(device)
        tactic_ids = torch.tensor([row["tactic_id"] for row in rows], device=device)
        hidden = self.initial_state(states, tactic_ids)
        feedback = hidden.new_zeros(count, self.hidden_dim)
        pools, memories = [], [{} for _ in rows]
        for row in rows:
            local = row["local_vectors"].to(device)
            library = self.lemma_adapter(row["library_vectors"].to(device))
            teacher = self.lemma_adapter(row["teacher_library_vectors"].to(device))
            pools.append({"local": local, "library": library, "teacher": teacher,
                "scores": row["library_scores"].to(device),
                "local_positions": {identity: pos for pos, identity in enumerate(row["local_ids"])},
                "library_positions": {identity: pos for pos, identity in enumerate(row["library_ids"])},
                "teacher_positions": {identity: pos for pos, identity in enumerate(row["teacher_library_ids"])}})
        stats = Counter(rows=count)
        hits = {key: hidden.new_zeros(()) for key in ("token_hits", "byte_hits", "structural_hits", "local_hits", "library_hits", "bound_hits")}
        row_exact = torch.ones(count, device=device, dtype=torch.bool)
        syntax_sum = hidden.new_zeros(count)
        pointer_losses = []
        for step in range(max(lengths)):
            hidden = self.advance(tokens[:, step], feedback, hidden)
            logits = self.token_head(hidden)
            logits[:, :3] = float("-inf")  # PAD/BOS/UNK are never action targets.
            target = tokens[:, step + 1]
            active = torch.tensor([step < length for length in lengths], device=device)
            # Padded rows use an arbitrary valid target; their loss is masked.
            safe_target = target.masked_fill(~active, 3)
            syntax_sum = syntax_sum + F.cross_entropy(logits, safe_target, reduction="none") * active
            correct = logits.argmax(dim=-1) == target
            byte = (target >= 3) & (target < 259) & active
            hits["token_hits"] += (correct & active).sum()
            hits["byte_hits"] += (correct & byte).sum()
            hits["structural_hits"] += (correct & active & ~byte).sum()
            stats["tokens"] += sum(step < length for length in lengths)
            stats["byte_tokens"] += sum(3 <= row["token_ids"][step + 1] < 259 for row in rows if step < len(row["token_ids"]) - 1)
            row_exact &= correct | ~active
            next_feedback = []
            for i, row in enumerate(rows):
                value = hidden.new_zeros(self.hidden_dim)
                if step >= lengths[i]:
                    next_feedback.append(value)
                    continue
                token = self.tokens[row["token_ids"][step + 1]]
                pool = pools[i]
                identity = row["pointer_targets"][step + 1]
                if token == "DECLARE_LEXICAL":
                    slot = row["declaration_slots"][step + 1]
                    if slot < 0 or slot in memories[i]:
                        raise ValueError("invalid lexical declaration in compiled prefix")
                    memories[i][slot] = self.lexical_key(hidden[i])
                elif token == "LOCAL":
                    allowed = row["input_local_scopes"][step + 1]
                    positions = [pool["local_positions"][key] for key in allowed]
                    if identity not in allowed:
                        raise ValueError("gold local is absent from legal input scope")
                    keys = pool["local"][positions]
                    scores = self.pointer_scores(hidden[i], keys)
                    chosen = allowed.index(identity)
                    pointer_losses.append(F.cross_entropy(scores.view(1, -1), torch.tensor([chosen], device=device)))
                    hit = scores.argmax() == chosen
                    hits["local_hits"] += hit
                    stats["local_positions"] += 1
                    row_exact[i] &= hit
                    value = keys[chosen]
                elif token == "LIBRARY":
                    stats["library_positions"] += 1
                    chosen = pool["library_positions"].get(identity)
                    if chosen is None:
                        stats["library_missing_positions"] += 1
                        row_exact[i] = False
                        value = pool["teacher"][pool["teacher_positions"][identity]]
                    else:
                        scores = (pool["scores"] * self.log_inverse_text_temperature.clamp(-4, 5).exp()
                                  + self.library_residual_scale.tanh() * self.pointer_scores(hidden[i], pool["library"]))
                        pointer_losses.append(F.cross_entropy(scores.view(1, -1), torch.tensor([chosen], device=device)))
                        hit = scores.argmax() == chosen
                        hits["library_hits"] += hit
                        stats["library_reachable_positions"] += 1
                        row_exact[i] &= hit
                        value = pool["library"][chosen]
                elif token == "BOUND":
                    allowed = row["lexical_scopes"][step + 1]
                    if identity not in allowed or any(key not in memories[i] for key in allowed):
                        raise ValueError("gold lexical pointer is outside decoded declaration scope")
                    keys = torch.stack([memories[i][key] for key in allowed])
                    scores = self.pointer_scores(hidden[i], keys)
                    chosen = allowed.index(identity)
                    pointer_losses.append(F.cross_entropy(scores.view(1, -1), torch.tensor([chosen], device=device)))
                    hit = scores.argmax() == chosen
                    hits["bound_hits"] += hit
                    stats["bound_positions"] += 1
                    row_exact[i] &= hit
                    value = keys[chosen]
                next_feedback.append(value)
            feedback = torch.stack(next_feedback)
        token_loss = (syntax_sum / torch.tensor(lengths, device=device)).mean()
        pointer_loss = torch.stack(pointer_losses).mean() if pointer_losses else token_loss.new_zeros(())
        loss = token_loss + pointer_loss
        stats.update({key: int(value.detach().item()) for key, value in hits.items()})
        stats["structural_tokens"] = stats["tokens"] - stats["byte_tokens"]
        stats["teacher_forced_exact_hits"] = int(row_exact.sum().item())
        stats["retrieval_reachable_rows"] = sum(row["library_positions_missing"] == 0 for row in rows)
        stats["known_tactics"] = sum(row["tactic_id"] != 0 for row in rows)
        stats["frozen_tactic_hits"] = sum(row["tactic_id"] != 0 and row["tactic_id"] == row["baseline_tactic_id"] for row in rows)
        return loss, {**dict(stats), "loss_sum": float(loss.detach()) * count,
                      "token_loss_sum": float(token_loss.detach()) * count,
                      "pointer_loss_sum": float(pointer_loss.detach()) * count}
