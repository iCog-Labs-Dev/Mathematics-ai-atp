from dataclasses import dataclass
from torch import Tensor
from .policy_actions import ArgumentReference


@dataclass
class ActionSample:
    tactic_action: Tensor
    tactic_logp: Tensor
    tactic_entropy: Tensor
    arguments: tuple[tuple[ArgumentReference, ...], ...]
    count_logp: Tensor
    source_logp: Tensor
    identity_logp: Tensor
    value: Tensor
    tactic_logits: Tensor

    @property
    def arg_logp(self):
        return self.count_logp + self.source_logp + self.identity_logp

    @property
    def logp(self):
        return self.tactic_logp + self.arg_logp
