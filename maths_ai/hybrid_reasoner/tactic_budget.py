"""One search's allowance of issued Lean tactic commands.

A step is one tactic command sent to Lean, whatever its outcome: accepted,
rejected or stopped by a resource limit. Exploration and combination replay
draw on the same allowance, because they run through the same executor. A
step is reserved before the command is sent, so the count never undercounts
work Lean may have done.
"""

from __future__ import annotations

__all__ = ["TacticAllowance", "TacticBudgetExhausted"]


class TacticBudgetExhausted(RuntimeError):
    """The search has issued every tactic command its allowance permits."""

    def __init__(self, limit: int) -> None:
        super().__init__(f"The tactic-step allowance of {limit} is used up.")
        self.limit = limit


class TacticAllowance:
    """A bounded counter of issued tactic commands."""

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit < 0:
            raise ValueError(f"A tactic allowance must be a nonnegative integer, got {limit!r}.")
        self.limit = limit
        self.issued = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.issued

    @property
    def exhausted(self) -> bool:
        return self.issued >= self.limit

    def reserve(self) -> None:
        """Count one command about to be issued, or refuse it when none remain."""
        if self.issued >= self.limit:
            raise TacticBudgetExhausted(self.limit)
        self.issued += 1

    def as_dict(self) -> dict[str, int]:
        return {"limit": self.limit, "issued": self.issued}
