"""Frozen GNN adapter using the same supervised operation seam as MCTS."""

from .backends import AttemptOutcome, BackendFailure, MctsFormalBackend, VerifyOutcome
from .admission import AdmissionRejected
PROFILE = "frozen-best-first"


class FrozenFormalBackend(MctsFormalBackend):
    name = "accepted-gnn"
    supports_universes = False

    def validate_deployment(self, deployment, profile):
        if profile.name != PROFILE or set(profile.backend_settings) - {"model_calls", "max_nodes", "max_depth", "top_k"}:
            raise ValueError("frozen runtime requires its explicit frozen-best-first profile")
        for key, value in profile.backend_settings.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"frozen profile {key} must be positive")
        loaded = (self.service.supervisor.provenance or {}).get("bundle_sha256")
        if deployment.model.get("bundle_sha256") != loaded:
            raise ValueError("frozen deployment does not bind the loaded accepted bundle")

    async def run(self, task, *, profile, selection, max_steps, timeout_s):
        view = await self._run("prove", {"manifest": task.manifest, "settings": dict(profile.backend_settings),
                                       "max_steps": max_steps}, timeout_s)
        return AttemptOutcome(**{key: value for key, value in view.items() if key in AttemptOutcome.__dataclass_fields__})

    async def verify(self, certificate, *, statement, max_steps, timeout_s):
        try:
            view = await self._run("verify", {"certificate": certificate, "statement": statement, "max_steps": max_steps}, timeout_s)
        except (TimeoutError, BackendFailure, AdmissionRejected) as exc:
            return VerifyOutcome("inconclusive", str(exc), None, None)
        return VerifyOutcome(**{key: value for key, value in view.items() if key in VerifyOutcome.__dataclass_fields__})
