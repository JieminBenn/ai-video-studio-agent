"""Tests for the orchestrator state machine — gates, resume, idempotency."""

import json

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.orchestrator.state_machine import StateMachine
from studio_agent.providers.base import ManualImportRequired
from studio_agent.providers.fake import FakeLLM
from studio_agent.creative_decisions import decision_preflight, resolve_decision
from studio_agent.stages.base import Providers, Stage, StageResult

PIPELINE = ["alpha", "beta"]


class CountingStage(Stage):
    """A stage that records how many times it actually ran (not skipped)."""

    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, project, providers):
        if project.stage_status(self.name) in ("complete", "approved"):
            return StageResult(status="skipped")
        self.runs += 1
        return StageResult(status="complete")


class AlwaysFailingStage(Stage):
    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, project, providers):
        self.runs += 1
        return StageResult(status="failed")


class RetryStage(Stage):
    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, project, providers):
        self.runs += 1
        return StageResult(status="complete" if self.runs > 1 else "failed")


class ManualImportStage(Stage):
    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, project, providers):
        self.runs += 1
        raise ManualImportRequired(
            provider="manual-midjourney",
            out_path=str(project.path("storyboard", "keyframes", "sh-001.png")),
            request_path=str(project.path("storyboard", "keyframes", "sh-001.png.midjourney.md")),
        )


class CrashingStage(Stage):
    def __init__(self, name):
        self.name = name
        self.runs = 0

    def run(self, project, providers):
        self.runs += 1
        if self.runs == 1:
            raise TypeError("structured provider output was invalid")
        return StageResult(status="complete")


class DecisionStage(CountingStage):
    def preflight(self, project, providers, *, auto=False):
        return decision_preflight(
            project,
            providers,
            stage=self.name,
            gap={
                "dimension": "camera_language",
                "default": "patient and intimate",
                "why": "Camera language controls every shot.",
            },
            auto=auto,
        )


class ApprovalRecordingStage(CountingStage):
    def __init__(self, name):
        super().__init__(name)
        self.approvals = []

    def on_approve(self, project, *, auto=False):
        self.approvals.append({"project_id": project.project_id, "auto": auto})


def _machine(tmp_path, idea="idea"):
    p = Project.create(idea, root=tmp_path, stages=PIPELINE)
    stages = [CountingStage("alpha"), CountingStage("beta")]
    return p, StateMachine(stages), stages


def test_run_executes_first_stage_then_pauses_at_gate(tmp_path):
    p, sm, stages = _machine(tmp_path)
    result = sm.run(p, Providers())

    assert result.paused_at == "alpha"
    assert p.stage_status("alpha") == "complete"
    assert stages[0].runs == 1
    assert stages[1].runs == 0  # never reached the second stage's gate
    assert p.current_stage == "alpha"


def test_approve_advances_to_next_stage(tmp_path):
    p, sm, _ = _machine(tmp_path)
    sm.run(p, Providers())
    sm.approve(p)
    assert p.stage_status("alpha") == "approved"
    assert p.current_stage == "beta"


def test_manual_approval_calls_stage_hook_once(tmp_path):
    project = Project.create("approval hook", root=tmp_path, stages=["alpha"])
    stage = ApprovalRecordingStage("alpha")
    machine = StateMachine([stage])
    machine.run(project, Providers())

    machine.approve(project)

    assert stage.approvals == [{"project_id": project.project_id, "auto": False}]


def test_approve_can_override_failed_stage_after_human_review(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha", "beta"])
    stage = AlwaysFailingStage("alpha")
    sm = StateMachine([stage, CountingStage("beta")])

    result = sm.run(p, Providers(), auto=True)
    new_stage = sm.approve(p)

    assert result.paused_at == "alpha"
    assert p.stage_status("alpha") == "approved"
    assert new_stage == "beta"
    assert p.current_stage == "beta"


def test_auto_runs_through_all_gates(tmp_path):
    p, sm, stages = _machine(tmp_path)
    result = sm.run(p, Providers(), auto=True)

    assert result.paused_at is None
    assert result.done is True
    assert stages[0].runs == 1 and stages[1].runs == 1
    assert p.status == "done"


def test_auto_approval_calls_stage_hook_once(tmp_path):
    project = Project.create("auto approval hook", root=tmp_path, stages=["alpha"])
    stage = ApprovalRecordingStage("alpha")

    result = StateMachine([stage]).run(project, Providers(), auto=True)

    assert result.done is True
    assert stage.approvals == [{"project_id": project.project_id, "auto": True}]


def test_resume_continues_without_rerunning_completed_stage(tmp_path):
    p, sm, stages = _machine(tmp_path)
    sm.run(p, Providers())     # runs alpha, pauses
    sm.approve(p)              # advance to beta

    # Resume: a fresh machine over the same project (simulating a restart).
    stages2 = [CountingStage("alpha"), CountingStage("beta")]
    sm2 = StateMachine(stages2)
    p2 = Project.load(p.dir)
    sm2.run(p2, Providers())

    assert stages2[0].runs == 0  # alpha not re-run on resume
    assert stages2[1].runs == 1  # beta runs
    assert p2.current_stage == "beta"


def test_completed_pipeline_run_is_noop(tmp_path):
    p, sm, stages = _machine(tmp_path)
    sm.run(p, Providers(), auto=True)
    total_before = stages[0].runs + stages[1].runs
    sm.run(p, Providers(), auto=True)
    assert stages[0].runs + stages[1].runs == total_before


def test_auto_resume_does_not_approve_failed_stage(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = AlwaysFailingStage("alpha")
    sm = StateMachine([stage])

    sm.run(p, Providers(), auto=True)
    result = sm.run(p, Providers(), auto=True)

    assert result.paused_at == "alpha"
    assert result.done is False
    assert p.stage_status("alpha") == "failed"
    assert p.current_stage == "alpha"
    assert stage.runs == 2


def test_failed_stage_can_be_retried_on_resume(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = RetryStage("alpha")
    sm = StateMachine([stage])

    first = sm.run(p, Providers(), auto=True)
    second = sm.run(p, Providers(), auto=True)

    assert first.paused_at == "alpha"
    assert second.done is True
    assert p.stage_status("alpha") == "approved"
    assert p.current_stage is None
    assert stage.runs == 2


def test_running_stage_is_resumable_after_interruption(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha"])
    p.set_stage_status("alpha", "running")
    p.save()
    stage = CountingStage("alpha")
    sm = StateMachine([stage])

    result = sm.run(p, Providers())

    assert result.paused_at == "alpha"
    assert p.stage_status("alpha") == "complete"
    assert stage.runs == 1


def test_failed_project_sync_stays_resumable(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha"])
    p.set_stage_status("alpha", "failed")
    p.save()

    reopened = Project.create("idea", root=tmp_path, stages=["alpha", "beta"])

    assert reopened.current_stage == "alpha"
    assert reopened.status == "in_progress"


def test_manual_import_required_pauses_stage_as_failed(tmp_path):
    p = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = ManualImportStage("alpha")
    sm = StateMachine([stage])

    result = sm.run(p, Providers(), auto=True)

    assert result.paused_at == "alpha"
    assert result.done is False
    assert result.manual_import_required is True
    assert result.request_path.endswith("sh-001.png.midjourney.md")
    assert p.stage_status("alpha") == "failed"
    assert p.current_stage == "alpha"
    assert stage.runs == 1


def test_unexpected_stage_exception_persists_failed_and_can_resume(tmp_path):
    p = Project.create("crash", root=tmp_path, stages=["alpha"])
    stage = CrashingStage("alpha")
    sm = StateMachine([stage])

    with pytest.raises(TypeError, match="structured provider output was invalid"):
        sm.run(p, Providers(), auto=True)

    persisted = Project.load(p.dir)
    assert persisted.stage_status("alpha") == "failed"
    assert persisted.current_stage == "alpha"

    result = sm.run(persisted, Providers(), auto=True)
    assert result.done is True
    assert persisted.stage_status("alpha") == "approved"


def test_review_mode_pauses_before_stage_runs_for_decision(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = DecisionStage("alpha")

    result = StateMachine([stage]).run(project, Providers(llm=FakeLLM()))

    assert result.decision_required is True
    assert stage.runs == 0
    assert project.stage_status("alpha") == "pending"
    assert result.request_path
    assert project.path("story", "decisions", "alpha.json").is_file()


def test_answered_decision_resumes_without_another_llm_call(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = DecisionStage("alpha")
    machine = StateMachine([stage])
    llm = FakeLLM()
    machine.run(project, Providers(llm=llm))
    cost_after_question = len(project.cost_log)
    resolve_decision(project, stage="alpha", choice="patient and intimate")

    result = machine.run(project, Providers(llm=llm), auto=True)

    assert result.done is True
    assert stage.runs == 1
    assert len(project.cost_log) == cost_after_question


def test_auto_records_default_and_runs_stage(tmp_path):
    project = Project.create("idea", root=tmp_path, stages=["alpha"])
    stage = DecisionStage("alpha")

    result = StateMachine([stage]).run(
        project, Providers(llm=FakeLLM()), auto=True
    )

    request = json.loads(
        project.path("story", "decisions", "alpha.json").read_text()
    )
    assert result.done is True
    assert request["resolution"]["source"] == "director_default"
    assert stage.runs == 1
