"""The orchestrator — a resumable state machine over the pipeline stages.

Each stage runs, writes its files, then the run **pauses at a human gate**
(invariant #2) unless ``auto=True``. ``project.json`` tracks ``current_stage`` and a
per-stage status so a re-run continues from the last incomplete stage and never
re-runs an already-completed one (invariant #3 — resumable + idempotent).

Per-stage status lifecycle::

    pending --run--> running --success--> complete --approve--> approved
       ^               |  |
       |               |  +--failure--> failed --approve (human override)--> approved
       +---------------+
       resume after interruption

``current_stage`` always points at the first stage that is not yet ``approved``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import ManualImportRequired
from ..stages.base import Providers, Stage
from .project import CostCapExceeded


@dataclass
class RunResult:
    paused_at: str | None = None  # stage awaiting human approval, or None
    done: bool = False            # whole pipeline approved
    cost_capped: bool = False     # halted to honor the per-project cost cap
    manual_import_required: bool = False
    decision_required: bool = False
    request_path: str | None = None
    message: str = ""


class StateMachine:
    def __init__(self, stages: list[Stage]):
        self.stages = stages
        self._by_name = {s.name: s for s in stages}

    def _next_stage(self, project, stage: str) -> str | None:
        idx = project.stages.index(stage)
        if idx + 1 < len(project.stages):
            return project.stages[idx + 1]
        return None

    def run(self, project, providers: Providers, *, auto: bool = False) -> RunResult:
        while project.current_stage is not None:
            stage_name = project.current_stage
            status = project.stage_status(stage_name)

            if status == "approved":
                project.current_stage = self._next_stage(project, stage_name)
                project.save()
                continue

            if status in ("pending", "failed", "running"):
                # Cost cap (invariant #7): don't start a paid stage with no budget left.
                if not project.within_cost_cap():
                    project.save()
                    return RunResult(paused_at=stage_name, cost_capped=True)
                try:
                    preflight = self._by_name[stage_name].preflight(
                        project, providers, auto=auto
                    )
                except CostCapExceeded:
                    project.save()
                    return RunResult(paused_at=stage_name, cost_capped=True)
                if preflight.decision_required:
                    project.save()
                    return RunResult(
                        paused_at=stage_name,
                        decision_required=True,
                        request_path=preflight.request_path,
                        message=preflight.message,
                    )
                try:
                    project.set_stage_status(stage_name, "running")
                    project.save()
                    result = self._by_name[stage_name].run(project, providers)
                except CostCapExceeded:
                    project.set_stage_status(stage_name, "failed")
                    project.save()
                    return RunResult(paused_at=stage_name, cost_capped=True)
                except ManualImportRequired as exc:
                    project.set_stage_status(stage_name, "failed")
                    project.save()
                    return RunResult(
                        paused_at=stage_name,
                        manual_import_required=True,
                        request_path=exc.request_path,
                        message=str(exc),
                    )
                except Exception:
                    project.set_stage_status(stage_name, "failed")
                    project.save()
                    raise
                if result.status == "failed":
                    project.set_stage_status(stage_name, "failed")
                    project.save()
                    return RunResult(paused_at=stage_name)
                project.set_stage_status(stage_name, "complete")
                project.save()
                # The stage logged its cost; stop before auto-advancing into more spend.
                if not project.within_cost_cap():
                    return RunResult(paused_at=stage_name, cost_capped=True)

            # status is now "complete" — at the human gate.
            if auto:
                self._by_name[stage_name].on_approve(project, auto=True)
                project.set_stage_status(stage_name, "approved")
                project.current_stage = self._next_stage(project, stage_name)
                project.save()
                continue
            return RunResult(paused_at=stage_name)

        project.status = "done"
        project.save()
        return RunResult(paused_at=None, done=True)

    def approve(self, project) -> str | None:
        """Pass the current gate; advance ``current_stage``. Returns the new stage."""
        stage_name = project.current_stage
        if stage_name is None:
            return None
        if project.stage_status(stage_name) in ("complete", "failed"):
            self._by_name[stage_name].on_approve(project, auto=False)
            project.set_stage_status(stage_name, "approved")
            project.current_stage = self._next_stage(project, stage_name)
            if project.current_stage is None:
                project.status = "done"
            project.save()
        return project.current_stage
