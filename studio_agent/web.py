"""Local web app for reviewing and steering file-backed projects.

This is intentionally dependency-free: it wraps the existing project files and CLI
orchestrator with a small browser surface so M1 can validate human gates before a
larger Next.js/SaaS UI exists.
"""

from __future__ import annotations

import html
import json
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.parse
import uuid
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .identity_board import coerce_board_dict, humanize_board_value
from .asset_regeneration import (
    delete_clip_shot,
    project_video_capabilities,
    regenerate_bible_asset,
    regenerate_bible_text,
    regenerate_keyframe,
    regenerate_shot_video,
    revise_keyframe_from_feedback,
    revise_video_from_feedback,
    set_clip_duration,
)
from .artifact_studio import apply_artifact_revision, mark_artifact_revised
from .creative_decisions import resolve_decision
from .shot_insertion import add_shot
from .invalidation import archive_paths, refresh_knowledge_target
from .orchestrator.project import Project
from .prompt_conversation import revise_prompt_turn
from .reference_assets import (
    load_reference_manifest,
    mark_reference_upload_revised,
    next_reference_alias,
    retarget_reference,
    save_reference_intake_uploads,
    save_reference_upload,
)
from .i18n import LANGS, UI_STRINGS, t
from .stage_workbench import resolve_stage, revision_target_for_stage, workbench_context

TEXT_EXTS = {".json", ".md", ".txt", ".yaml", ".yml"}
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".wav", ".mp3"}
CUSTOM_MODEL_PROFILE = "custom-model-mix"
MODEL_OPTION_KINDS = ("llm", "image", "video", "vlm")
MODEL_OPTION_META_KEYS = {
    "label", "group", "description", "requires", "provider", "availability", "capabilities",
}


@dataclass
class RunJob:
    id: str
    idea: str
    profile_name: str
    style_name: str | None
    format_name: str | None
    language: str
    auto: bool
    style_description: str = ""
    style_image: "UploadedFormFile | None" = None
    length: str = "auto"
    clip_plan: str = "default"
    clips: str = ""
    clip_seconds: str = ""
    model_parts: dict[str, str] = field(default_factory=dict)
    reference_uploads: list["UploadedFormFile"] = field(default_factory=list)
    reference_note: str = ""
    creative_inputs: dict[str, Any] = field(default_factory=dict)
    music_enabled: bool = False
    music_mood: str = ""
    kind: str = "create"
    status: str = "queued"
    project_id: str | None = None
    message: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    log: list[str] = field(default_factory=list)


@dataclass
class UploadedFormFile:
    filename: str
    content_type: str
    data: bytes


class ProjectBusyError(RuntimeError):
    """Raised when another dashboard action already owns the project writer lease."""


class JobRunner:
    """Small in-memory background runner for local dashboard actions."""

    def __init__(self, *, inline: bool = False):
        self.inline = inline
        self._lock = threading.Lock()
        self._jobs: dict[str, RunJob] = {}
        self._active_project_writers: set[str] = set()

    def start_run(
        self,
        root: Path,
        *,
        idea: str,
        profile_name: str,
        style_name: str | None = None,
        style_description: str = "",
        style_image: "UploadedFormFile | None" = None,
        format_name: str | None = None,
        language: str = "auto",
        auto: bool = True,
        length: str = "auto",
        clip_plan: str = "default",
        clips: str = "",
        clip_seconds: str = "",
        model_parts: dict[str, str] | None = None,
        reference_uploads: list[UploadedFormFile] | None = None,
        reference_note: str = "",
        creative_inputs: dict[str, Any] | None = None,
        music_enabled: bool = False,
        music_mood: str = "",
    ) -> RunJob:
        job = RunJob(
            id=uuid.uuid4().hex[:10],
            idea=idea,
            profile_name=profile_name,
            style_name=style_name,
            style_description=style_description,
            style_image=style_image,
            format_name=format_name,
            language=language,
            auto=auto,
            length=length,
            clip_plan=clip_plan,
            clips=clips,
            clip_seconds=clip_seconds,
            model_parts=dict(model_parts or {}),
            reference_uploads=list(reference_uploads or []),
            reference_note=reference_note,
            creative_inputs=dict(creative_inputs or {}),
            music_enabled=music_enabled,
            music_mood=music_mood,
        )
        with self._lock:
            self._jobs[job.id] = job
        return self._launch(root, job)

    def start_resume(
        self,
        root: Path,
        *,
        project_id: str,
        auto: bool = False,
    ) -> RunJob:
        """Run an existing project's pipeline in the background.

        Like the initial run, this returns immediately so the dashboard can
        poll project state and show the per-stage loading indicator while the
        next stage generates."""
        job = RunJob(
            id=uuid.uuid4().hex[:10],
            idea="",
            profile_name="",
            style_name=None,
            format_name=None,
            language="auto",
            auto=auto,
            kind="resume",
            project_id=project_id,
        )
        self._acquire_project_writer(project_id)
        try:
            with self._lock:
                self._jobs[job.id] = job
            return self._launch(root, job)
        except Exception:
            self._release_project_writer(project_id)
            raise

    def has_active_project_job(self, project_id: str) -> bool:
        """Return whether a queued/running job is still writing this project."""
        with self._lock:
            return project_id in self._active_project_writers or any(
                job.project_id == project_id and job.status in {"queued", "running"}
                for job in self._jobs.values()
            )

    @contextmanager
    def project_writer(self, project_id: str):
        self._acquire_project_writer(project_id)
        try:
            yield
        finally:
            self._release_project_writer(project_id)

    def _acquire_project_writer(self, project_id: str) -> None:
        with self._lock:
            if project_id in self._active_project_writers:
                raise ProjectBusyError(
                    f"project '{project_id}' already has an active job; wait for it to finish"
                )
            self._active_project_writers.add(project_id)

    def _release_project_writer(self, project_id: str) -> None:
        with self._lock:
            self._active_project_writers.discard(project_id)

    def _launch(self, root: Path, job: RunJob) -> RunJob:
        if self.inline:
            self._execute(root, job)
        else:
            thread = threading.Thread(target=self._execute, args=(Path(root), job), daemon=True)
            thread.start()
        return job

    def snapshot(self, root: Path) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [self._job_dict(job, Path(root)) for job in sorted(jobs, key=lambda j: -j.created_at)]

    def _set(self, job: RunJob, *, status: str | None = None,
             message: str | None = None, error: str | None = None,
             project_id: str | None = None) -> None:
        with self._lock:
            if status is not None:
                job.status = status
            if message is not None:
                job.message = message
                if message:
                    job.log.append(message)
                    job.log = job.log[-8:]
            if error is not None:
                job.error = error
            if project_id is not None:
                job.project_id = project_id
            job.updated_at = time.time()

    def _execute(self, root: Path, job: RunJob) -> None:
        if job.kind == "resume":
            self._execute_resume(root, job)
            return
        try:
            self._set(job, status="running", message="Creating project")
            project, result = _run_dashboard_project(
                root,
                job,
                progress=self._set,
                writer=self.project_writer,
            )
            status = "done" if result.done else "paused"
            message = (
                "Pipeline complete"
                if result.done else
                f"Cost cap reached at {result.paused_at}: ${project.total_cost():.2f} "
                f"spent of ${project.cost_cap}."
                if result.cost_capped else
                f"Manual import needed: {result.request_path}"
                if getattr(result, "manual_import_required", False) else
                f"Paused at {result.paused_at or project.current_stage or 'done'}"
            )
            self._set(job, status=status, message=message, project_id=project.project_id)
        except Exception as exc:  # local dashboard: surface actionable error text
            self._set(job, status="failed", error=str(exc), message="Run failed")

    def _execute_resume(self, root: Path, job: RunJob) -> None:
        from . import cli

        try:
            self._set(job, status="running", message="Resuming pipeline")
            project = Project.load(Path(root) / job.project_id)
            profile = project.model_config or cli.load_config()["profiles"]["fake"]
            result = cli._machine().run(
                project,
                cli.build_providers(profile),
                auto=job.auto,
            )
            status = "done" if result.done else "paused"
            message = (
                "Pipeline complete"
                if result.done else
                f"Cost cap reached at {result.paused_at}: ${project.total_cost():.2f} "
                f"spent of ${project.cost_cap}."
                if result.cost_capped else
                f"Manual import needed: {result.request_path}"
                if getattr(result, "manual_import_required", False) else
                f"Paused at {result.paused_at or project.current_stage or 'done'}"
            )
            self._set(job, status=status, message=message, project_id=project.project_id)
        except Exception as exc:  # local dashboard: surface actionable error text
            self._set(job, status="failed", error=str(exc), message="Resume failed")
        finally:
            if job.project_id:
                self._release_project_writer(job.project_id)

    def _job_dict(self, job: RunJob, root: Path) -> dict[str, Any]:
        project_status = None
        current_stage = None
        cost = 0.0
        if job.project_id and (root / job.project_id / "project.json").is_file():
            project = Project.load(root / job.project_id)
            project_status = project.status
            current_stage = project.current_stage
            cost = project.total_cost()
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "idea": job.idea,
            "profile": job.profile_name,
            "model_parts": dict(job.model_parts),
            "style": job.style_name,
            "format": job.format_name,
            "language": job.language,
            "auto": job.auto,
            "length": job.length,
            "creative_inputs": dict(job.creative_inputs),
            "project_id": job.project_id,
            "project_status": project_status,
            "current_stage": current_stage,
            "cost": cost,
            "message": job.message,
            "error": job.error,
            "log": list(job.log),
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }


def _run_dashboard_project(
    root: Path,
    job: RunJob,
    *,
    progress,
    writer=None,
) -> tuple[Project, Any]:
    """Create and run a project from dashboard fields using the same CLI contracts."""
    from . import cli

    config = cli.load_config()
    profile_name, profile, model_parts = _resolve_dashboard_model_config(config, job)
    # A free-form style description (or a style/global reference upload) defines a custom,
    # user-defined style the style stage profiles; otherwise fall back to a named preset.
    style_description = (job.style_description or "").strip()
    custom_style = bool(style_description) or job.style_image is not None or job.style_name is None
    try:
        resolved_style = cli.resolve_style(
            config, cli.CUSTOM_STYLE_NAME if custom_style else job.style_name
        )
        resolved_format = cli.resolve_format(config, job.format_name)
        language = cli.resolve_language(job.idea, job.language)
    except (cli.UnknownStyleError, cli.UnknownFormatError, cli.UnknownLanguageError) as exc:
        raise ValueError(str(exc)) from exc

    mode = cli.format_mode(resolved_format.spec)
    sized_spec = cli.apply_length(resolved_format.spec, mode, job.length)
    model_config = {
        "profile": profile_name,
        "style_name": resolved_style.name,
        "style": resolved_style.style,
        "format_name": resolved_format.name,
        "product_format": {"name": resolved_format.name, **sized_spec},
        "language": language,
        "clip_count": resolved_format.spec.get("clip_count", 1),
        "clip_seconds": resolved_format.spec.get("clip_seconds", 15),
        **profile,
        "audio_mode": cli.new_project_audio_mode(profile),
        "music_enabled": bool(job.music_enabled),
        "music_mood": (job.music_mood or "").strip(),
        "motion_grid": config.get("motion_grid", {"enabled": False, "layout": "auto"}),
        "storyboard": config.get("storyboard", {"flow_review": {"enabled": True}}),
    }
    if mode == "clip":
        model_config["clip_target_duration_s"] = sized_spec["clip_target_duration_s"]
        plan = str(job.clip_plan or "").strip().lower()
        if plan == "manual":
            model_config["clip_plan_mode"] = "manual"
            model_config["clip_count"] = _positive_int_or(job.clips, 1)
            model_config["clip_seconds"] = _positive_int_or(job.clip_seconds, None)
        elif plan == "auto":
            model_config["clip_plan_mode"] = "auto"
        else:
            model_config["clip_plan_mode"] = "default"
    if model_parts:
        model_config["model_parts"] = model_parts
    model_config["creative_inputs"] = dict(job.creative_inputs)
    if custom_style:
        model_config["style_input"] = {"description": style_description}

    existing_id = next(
        (
            item["id"]
            for item in project_summaries(Path(root))
            if item.get("idea") == job.idea
        ),
        None,
    )
    with ExitStack() as stack:
        if writer is not None and existing_id:
            stack.enter_context(writer(existing_id))
        project = Project.create(
            job.idea,
            root=root,
            stages=cli.pipeline_for(resolved_format),
            cost_cap=config.get("cost_cap"),
            model_config=model_config,
        )
        if writer is not None and not existing_id:
            stack.enter_context(writer(project.project_id))
        project.story_dir.joinpath("idea.md").write_text(job.idea + "\n")
        project.save()
        progress(job, project_id=project.project_id, message=f"Project created: {project.project_id}")

        providers = cli.build_providers(profile)
        if job.style_image is not None:
            save_reference_upload(
                project,
                target_type="style",
                target_id="global",
                data=job.style_image.data,
                filename=job.style_image.filename,
                content_type=job.style_image.content_type,
                label="style reference",
            )
            progress(job, message="Saved style reference image")
        if job.reference_uploads:
            records = save_reference_intake_uploads(
                project,
                job.reference_uploads,
                note=job.reference_note,
                analyzer=providers.reference_analyzer,
            )
            progress(job, message=f"Saved {len(records)} starting reference(s)")
        progress(job, message=f"Running pipeline with {profile_name}")
        result = cli._machine().run(project, providers, auto=job.auto)
        return project, result


def _resolve_dashboard_model_config(
    config: dict,
    job: RunJob,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    if job.model_parts:
        return (
            CUSTOM_MODEL_PROFILE,
            _compose_model_mix(config, job.model_parts),
            _selected_model_parts(config, job.model_parts),
        )
    profiles = config.get("profiles", {})
    if job.profile_name not in profiles:
        raise ValueError(
            f"unknown profile '{job.profile_name}'. Available: {', '.join(profiles)}"
        )
    return job.profile_name, dict(profiles[job.profile_name]), {}


def _compose_model_mix(config: dict, requested: dict[str, str]) -> dict[str, Any]:
    """Merge independently selected provider components into one runtime profile."""
    selected = _selected_model_parts(config, requested)
    profile: dict[str, Any] = {
        "tts": "fake",
        "music": "fake",
        "audio_mode": "native_video",
    }
    for kind in MODEL_OPTION_KINDS:
        option = _model_option(config, kind, selected[kind])
        for key, value in option.items():
            if key not in MODEL_OPTION_META_KEYS:
                profile[key] = value
    return profile


def _selected_model_parts(config: dict, requested: dict[str, str]) -> dict[str, str]:
    defaults = config.get("default_model_options", {})
    selected = {}
    for kind in MODEL_OPTION_KINDS:
        value = requested.get(kind) or defaults.get(kind) or "fake"
        _model_option(config, kind, value)
        selected[kind] = value
    return selected


def _model_option(config: dict, kind: str, option_name: str) -> dict[str, Any]:
    options = config.get("model_options", {}).get(kind, {})
    if option_name not in options:
        available = ", ".join(options) or "none"
        raise ValueError(
            f"unknown {kind} model option '{option_name}'. Available: {available}"
        )
    return dict(options[option_name] or {})


def apply_video_model(config: dict, model_config: dict, choice: str) -> dict:
    """Return a copy of ``model_config`` with only the video component swapped to ``choice``.

    Strips every existing ``video*`` key, overlays the chosen ``model_options.video`` entry's
    non-meta keys (all ``video``/``video_*``), and records ``model_parts['video'] = choice``.
    Leaves llm/image/vlm/tts/music/audio/style untouched. Raises ``ValueError`` on an unknown
    choice (via ``_model_option``). Each catalog entry is self-contained, so strip-then-overlay
    fully replaces the previous model's video settings.
    """
    option = _model_option(config, "video", choice)
    updated = {k: v for k, v in model_config.items() if not k.startswith("video")}
    for key, value in option.items():
        if key not in MODEL_OPTION_META_KEYS:
            updated[key] = value
    parts = dict(updated.get("model_parts") or {})
    parts["video"] = choice
    updated["model_parts"] = parts
    return updated


def video_model_key_missing(config: dict, choice: str) -> str | None:
    """Return the first required API-key env var that is unset for ``choice``, else ``None``.

    Uses the same requirement resolution as the dropdown label
    (``_model_option_requirements``) so the guardrail and the label never disagree —
    including providers (fal, volcengine) that declare their key via the provider
    fallback map rather than an explicit ``video_api_key_env``. ``fake`` needs no key.
    """
    option = _model_option(config, "video", choice)
    for env in _model_option_requirements("video", option):
        if env and not os.environ.get(env):
            return env
    return None


def current_video_model(config: dict, model_config: dict) -> str:
    """The catalog name of the project's current video model, for the picker default."""
    catalog = (config.get("model_options", {}) or {}).get("video", {}) or {}
    parts = model_config.get("model_parts") or {}
    if parts.get("video") in catalog:
        return parts["video"]
    for name, option in catalog.items():
        if (option.get("video") == model_config.get("video")
                and option.get("video_model", "") == model_config.get("video_model", "")):
            return name
    return (config.get("default_model_options", {}) or {}).get("video", "fake")


def maybe_apply_video_model(project: "Project", config: dict, fields: dict) -> str | None:
    """Swap+persist the project's video model if the regenerate form carried a new choice.

    Returns a guardrail notice (changing nothing) when the chosen model's API key is unset;
    returns ``None`` after persisting a valid change, or when the choice is absent/unchanged.
    """
    choice = (fields.get("video_model") or [""])[0].strip()
    if not choice or choice == current_video_model(config, project.model_config or {}):
        return None
    missing = video_model_key_missing(config, choice)
    if missing:
        return f"Set {missing} to use this video model."
    project.model_config = apply_video_model(config, project.model_config or {}, choice)
    project.save()
    return None


_JOBS = JobRunner()


def project_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    if not root.is_dir():
        return summaries
    for child in sorted(root.iterdir()):
        if not (child / "project.json").is_file():
            continue
        project = Project.load(child)
        summaries.append(
            {
                "id": project.project_id,
                "idea": project.idea,
                "status": project.status,
                "current_stage": project.current_stage,
                "cost": project.total_cost(),
                "profile": project.model_config.get("profile"),
                "format": project.model_config.get("format_name"),
                "language": project.model_config.get("language"),
            }
        )
    return summaries


def project_artifacts(project: Project) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(project.dir.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        rel = path.relative_to(project.dir).as_posix()
        suffix = path.suffix.lower()
        artifacts.append(
            {
                "path": rel,
                "kind": "text" if suffix in TEXT_EXTS else "media" if suffix in MEDIA_EXTS else "file",
                "size": path.stat().st_size,
            }
        )
    return artifacts


def project_state(project: Project) -> dict[str, Any]:
    return {
        "id": project.project_id,
        "status": project.status,
        "current_stage": project.current_stage,
        "stage_statuses": dict(project.stage_statuses),
        "cost": project.total_cost(),
        "version": _project_version(project),
    }


def shot_reviews(project: Project) -> list[dict[str, Any]]:
    """Per-shot media rows for the manual human review gate."""
    shots_path = project.path("storyboard", "shots.json")
    if not shots_path.is_file():
        return []
    shots = json.loads(shots_path.read_text()).get("shots", [])

    reviews = []
    for shot in shots:
        sid = shot["id"]
        clip = project.path("assets", "clips", f"{sid}.mp4")
        keyframe = project.path("storyboard", "keyframes", shot.get("keyframe", ""))
        video_prompt = project.path("storyboard", "prompts", f"{sid}.video.md")
        reviews.append({
            "id": sid,
            "clip": f"assets/clips/{sid}.mp4" if clip.is_file() else None,
            "keyframe": (
                f"storyboard/keyframes/{shot.get('keyframe')}" if keyframe.is_file() else None
            ),
            "duration_s": shot.get("duration_s"),
            # Surface the motion the clip was rendered from, so the human can see the 运镜
            # they're reviewing (and steering with the revise box) — not just the clip.
            "camera": shot.get("camera") or "",
            "camera_movement": shot.get("camera_movement") or "",
            "video_prompt": (
                f"storyboard/prompts/{sid}.video.md" if video_prompt.is_file() else None
            ),
            "review_mode": "manual",
        })
    return reviews


def _render_shot_card(project: Project, review: dict[str, Any], *, lang: str = "en") -> str:
    sid = review["id"]
    media = lambda rel: f"/projects/{_u(project.project_id)}/media?path={_u(rel)}"

    if review["clip"]:
        poster = f" poster='{media(review['keyframe'])}'" if review["keyframe"] else ""
        clip_html = f"<video controls preload='metadata'{poster} src='{media(review['clip'])}'></video>"
    elif review["keyframe"]:
        clip_html = f"<img src='{media(review['keyframe'])}' alt='{_h(sid)} keyframe'>"
    else:
        clip_html = f"<p class='muted'>{t('No clip yet.', lang)}</p>"

    badge = f"<span class='badge none'>{t('MANUAL REVIEW', lang)}</span>"
    detail = (
        f"<p class='muted'>{t('Compare the clip with its keyframe, then approve, edit the prompt, or regenerate this shot.', lang)}</p>"
    )

    # Show the motion (运镜) this clip was rendered from — the camera move inline plus a link
    # to the full <id>.video.md — so the human can see what they're reviewing and revising.
    motion = ""
    movement = " · ".join(p for p in [review.get("camera"), review.get("camera_movement")] if p)
    if movement:
        motion += f"<p class='motion'><b>{t('Camera movement', lang)}</b>: {_h(movement)}</p>"
    if review.get("video_prompt"):
        motion += _artifact_link(project, review["video_prompt"], t("View motion prompt", lang))

    # Steer the LLM on the motion before spending on a re-render: revise <id>.video.md from a
    # comment (cheap, no paid run), then click Regenerate to re-render from the revised prompt.
    revise = ""
    if review["clip"]:
        revise = (
            f"<form class='video-feedback' method='post' "
            f"action='/projects/{_u(project.project_id)}/video-feedback' "
            f"data-busy-message='Revising motion prompt for {_h(sid)}...'>"
            f"<input type='hidden' name='shot_id' value='{_h(sid)}'>"
            f"<label>{t('Feedback', lang)}"
            "<textarea name='feedback' required "
            "placeholder='Slow the camera push, hold on her face longer, or start the move later.'>"
            "</textarea></label>"
            f"<button type='submit' class='secondary'>{_h(t('Revise motion prompt', lang))}</button>"
            "</form>"
        )

    _cfg = _safe_config()
    _current_video = current_video_model(_cfg, project.model_config or {})
    _video_select = _model_option_select(_cfg, "video", selected=_current_video)
    regen = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/regenerate' "
        f"data-busy-message='Regenerating {_h(sid)}...'>"
        f"<input type='hidden' name='shot' value='{_h(sid)}'>"
        f"<label>{_h(t('Video model', lang))}"
        f"<select name='video_model'>{_video_select}</select></label>"
        f"<button type='submit'>{t('Regenerate this shot', lang)}</button></form>"
    )
    structure = (
        "<div class='shot-structure'>"
        + _delete_shot_form(project, sid, stage="video", lang=lang)
        + _add_shot_form(project, sid, lang=lang)
        + "</div>"
    )
    return (
        f"<article class='shot'><header><b>{_h(sid)}</b>{badge}</header>"
        f"{clip_html}{detail}{motion}{revise}{regen}{structure}</article>"
    )


def safe_project_path(project: Project, rel_path: str) -> Path:
    rel = Path(urllib.parse.unquote(rel_path))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("artifact path must stay inside the project")
    path = (project.dir / rel).resolve()
    root = project.dir.resolve()
    if root != path and root not in path.parents:
        raise ValueError("artifact path must stay inside the project")
    return path


def delete_project(root: Path, project_id: str) -> None:
    """Permanently remove one validated project directory."""
    root = Path(root).resolve()
    if not project_id or project_id in {".", ".."} or Path(project_id).name != project_id:
        raise ValueError("invalid project id")

    candidate = root / project_id
    try:
        project_dir = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"project not found: {project_id}") from exc
    if candidate.is_symlink() or project_dir.parent != root:
        raise ValueError("invalid project id")
    if not project_dir.is_dir() or not project_dir.joinpath("project.json").is_file():
        raise FileNotFoundError(f"project not found: {project_id}")

    shutil.rmtree(project_dir)


def render_index(root: Path, jobs: JobRunner | None = None, notice: str = "", *, lang: str = "en") -> str:
    rows = []
    for project in project_summaries(root):
        rows.append(
            "<tr>"
            f"<td><a class='mono' href='/projects/{_u(project['id'])}'>{_h(project['id'])}</a></td>"
            f"<td>{_h(project['idea'])}</td>"
            f"<td><span class='badge {_status_class(project['status'])}'>{_h(project['status'])}</span></td>"
            f"<td class='mono'>{_h(project['current_stage'] or 'done')}</td>"
            f"<td>{_h(project.get('format') or '')}</td>"
            f"<td>{_h(project.get('language') or '')}</td>"
            f"<td class='mono'>${project['cost']:.2f}</td>"
            "<td class='project-actions'>"
            f"<button type='button' class='delete-trigger' data-delete-project "
            f"data-project-id='{_h(project['id'])}' aria-haspopup='dialog'>{t('Delete', lang)}</button>"
            "</td>"
            "</tr>"
        )
    config = _safe_config()
    dashboard = _render_dashboard(root, config, jobs or _JOBS, lang=lang)
    body = (
        "<header class='toolbar'><span class='brand'>Studio Agent</span>"
        f"<span class='sub'>{t('Local production console', lang)}</span>"
        f"{_language_toggle(lang)}</header>"
        f"{'<p class=notice>' + _h(notice) + '</p>' if notice else ''}"
        "<main class='dashboard'>"
        f"{dashboard}"
        f"<section id='projects' class='wide projects-panel'><h2>{t('Projects', lang)}</h2><div class='table-scroll'>"
        f"<table><thead><tr><th>{t('Project', lang)}</th><th>{t('Idea', lang)}</th><th>{t('Status', lang)}</th>"
        f"<th>{t('Gate', lang)}</th><th>Format</th><th>Lang</th><th>{t('Cost', lang)}</th><th>{t('Actions', lang)}</th></tr></thead>"
        f"<tbody>{''.join(rows) or '<tr><td colspan=8 class=muted>No projects yet.</td></tr>'}</tbody>"
        "</table></div></section></main>"
        "<dialog class='delete-dialog' data-delete-dialog aria-labelledby='delete-dialog-title'>"
        f"<form method='post' action='/' data-delete-form data-busy-message='{_h(t('Deleting project...', lang))}'>"
        "<div class='delete-dialog-mark' aria-hidden='true'>!</div>"
        f"<div><span class='delete-dialog-kicker'>{t('Destructive action', lang)}</span>"
        f"<h3 id='delete-dialog-title'>{t('Permanently delete project?', lang)}</h3></div>"
        f"<p class='delete-dialog-warning'>{t('This permanently removes every generated file for:', lang)}</p>"
        "<p class='delete-dialog-project mono' data-delete-project-name></p>"
        f"<p class='muted'>{t('This cannot be undone.', lang)}</p>"
        "<div class='delete-dialog-actions'>"
        f"<button type='button' class='secondary' data-delete-cancel>{t('Cancel', lang)}</button>"
        f"<button type='submit' class='delete-confirm'>{t('Delete project', lang)}</button>"
        "</div></form></dialog>"
    )
    return page("Projects", body, lang=lang)


def _safe_config() -> dict:
    try:
        from . import cli
        return cli.load_config()
    except Exception:
        return {"profiles": {"fake": {}}, "style_presets": {}, "product_formats": {}}


def _revision_providers(project: Project):
    """Build providers for lightweight LLM revision actions.

    The dashboard can render and apply file edits even when the local Python used for
    a smoke test does not have the full CLI config dependencies installed. Real
    configured providers still go through ``cli.build_providers``; the fallback is
    only for fake/no-profile revision work.
    """
    try:
        from . import cli
        profile = project.model_config or cli.load_config()["profiles"]["fake"]
        return cli.build_providers(profile)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        profile = project.model_config or {}
        if profile.get("llm") not in (None, "fake"):
            raise
        from .providers.fake import FakeLLM, FakeReferenceAnalyzer
        from .stages.base import Providers
        return Providers(llm=FakeLLM(), reference_analyzer=FakeReferenceAnalyzer())


def _active_prompt_kind(project: Project) -> str:
    if project.current_stage in {"clip", "storyboard"}:
        return "keyframes"
    if project.current_stage == "video_prompts":
        return "videos"
    return ""


def _positive_int_or(value, default):
    """Parse a form number field; blank/invalid/non-positive → ``default``."""
    text = str(value or "").strip()
    if not text:
        return default
    try:
        number = int(round(float(text)))
    except ValueError:
        return default
    return number if number > 0 else default


def _render_dashboard(root: Path, config: dict, jobs: JobRunner, *, lang: str = "en") -> str:
    default_parts = config.get("default_model_options", {})
    llm_options = _model_option_select(config, "llm", selected=default_parts.get("llm"))
    image_options = _model_option_select(config, "image", selected=default_parts.get("image"))
    video_options = _model_option_select(config, "video", selected=default_parts.get("video"))
    vlm_options = _model_option_select(config, "vlm", selected=default_parts.get("vlm"))
    product_formats = config.get("product_formats", {})
    format_options = _format_options(
        product_formats,
        selected=config.get("default_format", "short_film"),
        include_blank=False,
    )
    # Map each visible format to its pipeline mode so the client can switch the
    # length unit (clip → seconds, story/film → minutes).
    format_modes = {
        key: (spec.get("mode") or "story")
        for key, spec in product_formats.items()
        if not spec.get("ui_hidden")
    }
    # Render the initial visibility server-side so the clip-plan fields only appear for
    # a clip-mode default format (the JS toggle keeps them in sync on format change).
    default_mode = format_modes.get(config.get("default_format", "short_film"), "story")
    clip_plan_hidden = "" if default_mode == "clip" else " hidden"
    length_hidden = " hidden" if default_mode == "clip" else ""
    job_rows = _render_job_rows(jobs.snapshot(root), lang=lang)
    _idea_placeholder = _h(t("A princess stepping from a glowing television into a man's home", lang))
    return (
        f"<section class='run-panel wide'><div><h2>{t('New Run', lang)}</h2>"
        f"<p class='muted'>{t('Choose each stage model independently; every cross-provider combination is allowed.', lang)}</p></div>"
        "<form method='post' action='/runs' class='run-form' enctype='multipart/form-data'>"
        f"<label class='full'>{t('Idea', lang)}<textarea name='idea' required "
        f"placeholder='{_idea_placeholder}'></textarea></label>"
        f"<div class='model-mix full'><h3>{t('Creative direction (optional)', lang)}</h3>"
        f"<p class='muted'>{t('Lock what matters now; the director asks only about unresolved choices later.', lang)}</p>"
        "<div class='model-mix-grid'>"
        f"<label>{t('Tone', lang)}<input name='tone' placeholder='tender but uneasy'></label>"
        f"<label>{t('Visual world', lang)}<input name='visual_world' placeholder='tactile coastal realism'></label>"
        f"<label>{t('Camera language', lang)}<input name='camera_language' placeholder='patient observation, then move closer'></label>"
        f"<label>{_h(t('Light & texture', lang))}<input name='light_texture' placeholder='soft dawn haze, weathered surfaces'></label>"
        f"<label>{t('Character treatment', lang)}<input name='character_treatment' placeholder='restrained acting, stable silhouette'></label>"
        f"</div><label class='full'>{t('Hard avoidances', lang)}<textarea name='hard_avoidances' "
        "placeholder='one rule per line, e.g. unmotivated orbit'></textarea></label></div>"
        f"{_reference_picker(t('Reference photos', lang), 'reference_note', lang=lang)}"
        f"<label class='full'>{t('What are these for?', lang)}"
        "<textarea name='reference_note' "
        "placeholder='@image1 is the protagonist; @image2 is the apartment; @image1 kisses @image2'></textarea></label>"
        f"<div class='model-mix full'><h3>{t('Model selection', lang)}</h3>"
        f"<p class='muted'>{t('Pick each model below; the run uses exactly what you choose.', lang)}</p>"
        "<div class='model-mix-grid'>"
        f"<label>{t('Story / script LLM', lang)}<select name='llm_option' data-model-preference='llm'>{llm_options}</select></label>"
        f"<label>{t('Image / keyframes', lang)}<select name='image_option' data-model-preference='image'>{image_options}</select></label>"
        f"<label>{t('Video generation', lang)}<select name='video_option' data-model-preference='video'>{video_options}</select></label>"
        f"<label>{_h(t('Vision model (reference & style)', lang))}<select name='vlm_option' data-model-preference='vlm'>{vlm_options}</select></label>"
        "</div></div>"
        f"<div class='model-mix full'><h3>{t('Style', lang)}</h3>"
        f"<p class='muted'>{t('Define one look for the whole film/series — describe it, upload a reference image (only its style is used, never its subject), or both.', lang)}</p>"
        f"<label class='full'>{t('Describe your style', lang)}"
        "<input name='style_description' "
        "placeholder='e.g. 1970s grainy analog sci-fi, amber and teal'></label>"
        f"<label class='full'>{t('Style reference image', lang)}"
        "<input type='file' name='style_image' data-style-file "
        "accept='image/png,image/jpeg,image/webp,image/gif'></label>"
        "<div class='reference-preview' data-style-preview aria-live='polite' hidden></div>"
        "</div>"
        f"<label>{t('Format', lang)}<select name='format' data-format-select>{format_options}</select></label>"
        f"<label data-length-label{length_hidden}>{t('Length', lang)}"
        "<span class='length-field'>"
        f"<input type='number' name='length' min='1' step='1' placeholder='{_h(t('Auto', lang))}'>"
        f"<span class='length-unit' data-length-unit>{t('min', lang)}</span>"
        "</span></label>"
        f"<div class='model-mix full' data-clip-plan{clip_plan_hidden}>"
        f"<h3>{t('Clips', lang)}</h3>"
        f"<p class='muted'>{t('Default is one clip at the video model’s default length.', lang)}</p>"
        "<div class='model-mix-grid'>"
        f"<label>{t('Clip plan', lang)}<select name='clip_plan' data-clip-plan-select>"
        f"<option value='default'>{t('One clip (model default length)', lang)}</option>"
        f"<option value='manual'>{_h(t('Custom count & length', lang))}</option>"
        f"<option value='auto'>{t('Auto (AI plans the clips)', lang)}</option>"
        "</select></label>"
        f"<label>{t('Number of clips', lang)}"
        "<input type='number' name='clips' min='1' step='1' placeholder='1' data-clip-count></label>"
        f"<label>{t('Seconds per clip', lang)}"
        f"<input type='number' name='clip_seconds' min='1' step='1' "
        f"placeholder='{_h(t('Model default', lang))}' data-clip-seconds></label>"
        "</div></div>"
        "<label class='checkbox'><input type='checkbox' name='music'> "
        f"{_h(t('Background music', lang))}</label>"
        f"<label>{_h(t('Music mood (optional)', lang))}"
        "<input name='music_mood' placeholder='warm, sparse piano'></label>"
        "<div class='run-mode full'>"
        "<label><input type='radio' name='run_mode' value='review' checked>"
        f"<span>{t('Review each stage', lang)}</span></label>"
        "<label><input type='radio' name='run_mode' value='auto'>"
        f"<span>{t('Auto approve', lang)}</span></label>"
        "</div>"
        "<div class='run-actions full'>"
        f"<button type='submit' class='danger'>{t('Run full pipeline', lang)}</button>"
        "</div>"
        "</form></section>"
        f"<section class='wide'><div class='section-head'><h2>{t('Job Monitor', lang)}</h2>"
        f"<span class='muted'>{t('Auto-refreshes while this page is open.', lang)}</span></div>"
        f"<div id='jobs' class='jobs'>{job_rows}</div></section>"
        "<script>"
        "async function refreshJobs(){try{const r=await fetch('/jobs.json');"
        "const d=await r.json();document.getElementById('jobs').innerHTML=d.html;}catch(e){}}"
        "setInterval(refreshJobs,3000);"
        f"var STUDIO_FORMAT_MODES={json.dumps(format_modes)};"
        "function studioSyncLengthUnit(){"
        "var sel=document.querySelector('[data-format-select]');"
        "var unit=document.querySelector('[data-length-unit]');"
        "if(!sel||!unit)return;"
        "var mode=STUDIO_FORMAT_MODES[sel.value]||'story';"
        "unit.textContent=studioT((mode==='clip')?'sec':'min');"
        "var lengthLabel=document.querySelector('[data-length-label]');"
        "var clipPlan=document.querySelector('[data-clip-plan]');"
        "if(lengthLabel)lengthLabel.hidden=(mode==='clip');"
        "if(clipPlan)clipPlan.hidden=(mode!=='clip');"
        "studioSyncClipPlan();}"
        "function studioSyncClipPlan(){"
        "var plan=document.querySelector('[data-clip-plan-select]');"
        "var count=document.querySelector('[data-clip-count]');"
        "var secs=document.querySelector('[data-clip-seconds]');"
        "if(!plan||!count||!secs)return;"
        "var manual=(plan.value==='manual');"
        "count.disabled=!manual;secs.disabled=!manual;}"
        "document.addEventListener('change',function(e){"
        "if(!e.target||!e.target.getAttribute)return;"
        "if(e.target.getAttribute('data-format-select')!==null)studioSyncLengthUnit();"
        "if(e.target.getAttribute('data-clip-plan-select')!==null)studioSyncClipPlan();});"
        "studioSyncLengthUnit();"
        "</script>"
    )


def _reference_picker(
    label: str,
    note_name: str,
    *,
    alias_start: int | str = 1,
    required: bool = False,
    lang: str = "en",
) -> str:
    required_attr = " required" if required else ""
    return (
        "<div class='reference-picker full' data-reference-picker "
        f"data-reference-note='{_h(note_name)}' data-alias-start='{_h(alias_start)}'>"
        f"<label>{_h(label)}"
        "<input type='file' name='reference_images' data-reference-files "
        "accept='image/png,image/jpeg,image/webp,image/gif' multiple"
        f"{required_attr}></label>"
        "<div class='reference-preview' data-reference-preview aria-live='polite' hidden></div>"
        f"<p class='reference-picker-hint'>{t('Click an @image label to insert it into your note.', lang)}</p>"
        "</div>"
    )


def _model_option_select(config: dict, kind: str, *, selected: str | None = None) -> str:
    options = config.get("model_options", {}).get(kind, {})
    if not options:
        return "<option value='fake'>Fake</option>"
    html_parts: list[str] = []
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, option in options.items():
        group = str(option.get("group") or "Other")
        groups.setdefault(group, []).append((name, option))
    for group, items in groups.items():
        html_parts.append(f"<optgroup label='{_h(group)}'>")
        for name, option in items:
            sel = " selected" if name == selected else ""
            html_parts.append(
                f"<option value='{_h(name)}'{sel}>{_h(_model_option_label(kind, name, option))}</option>"
            )
        html_parts.append("</optgroup>")
    return "".join(html_parts)


def _model_option_label(kind: str, name: str, option: dict[str, Any]) -> str:
    label = str(option.get("label") or name)
    model_key = {
        "llm": "llm_model",
        "image": "image_model",
        "video": "video_model",
        "vlm": "vlm_model",
    }.get(kind)
    model = option.get(model_key or "")
    parts = [f"{label} — {model}" if model else label]
    availability = str(option.get("availability") or "stable").strip().lower()
    if availability not in ("", "stable", "ga"):
        parts.append(availability)
    requires = _model_option_requirements(kind, option)
    if isinstance(requires, str):
        requires = [requires]
    if requires:
        missing = [str(key) for key in requires if not os.environ.get(str(key))]
        parts.append(f"needs {', '.join(missing)}" if missing else "ready")
    return " · ".join(parts)


def _model_option_requirements(kind: str, option: dict[str, Any]) -> list[str]:
    configured = option.get("requires") or []
    if isinstance(configured, str):
        configured = [configured]
    if configured:
        return [str(key) for key in configured]
    explicit = option.get(f"{kind}_api_key_env")
    if explicit:
        return [str(explicit)]
    provider = str(option.get(kind) or "")
    defaults = {
        ("image", "gemini"): "GEMINI_API_KEY",
        ("image", "imagen"): "GEMINI_API_KEY",
        ("image", "openai"): "OPENAI_API_KEY",
        ("image", "xai"): "XAI_API_KEY",
        ("video", "fal"): "FAL_KEY",
        ("video", "volcengine"): "ARK_API_KEY",
        ("video", "byteplus"): "BYTEPLUS_ARK_API_KEY",
        ("video", "gemini-veo"): "GEMINI_API_KEY",
        ("video", "xai"): "XAI_API_KEY",
        ("vlm", "gemini"): "GEMINI_API_KEY",
        ("vlm", "anthropic"): "ANTHROPIC_API_KEY",
    }
    key = defaults.get((kind, provider))
    return [key] if key else []


def _render_job_rows(items: list[dict[str, Any]], *, lang: str = "en") -> str:
    if not items:
        return f"<p class='muted'>{t('No jobs yet.', lang)}</p>"
    rows = []
    for item in items:
        project = item.get("project_id")
        project_link = (
            f"<a href='/projects/{_u(project)}'>{_h(project)}</a>" if project else f"<span class='muted'>{t('pending', lang)}</span>"
        )
        status = _h(item.get("status"))
        log = " / ".join(item.get("log") or [])
        detail = item.get("error") or item.get("message") or log
        parts = item.get("model_parts") or {}
        model_mix = (
            " · ".join(f"{kind}:{value}" for kind, value in parts.items())
            if parts else ""
        )
        rows.append(
            "<article class='job'>"
            f"<header><b>{t('Model mix', lang)}</b><span class='badge {status}'>{t(status.upper(), lang)}</span></header>"
            f"<p>{_h(item.get('idea'))}</p>"
            "<dl>"
            f"<dt>{t('Project', lang)}</dt><dd>{project_link}</dd>"
            f"<dt>{t('Stage', lang)}</dt><dd>{_h(item.get('current_stage') or item.get('project_status') or 'pending')}</dd>"
            f"<dt>{t('Cost', lang)}</dt><dd>${float(item.get('cost') or 0):.2f}</dd>"
            f"{'<dt>' + t('Models', lang) + '</dt><dd>' + _h(model_mix) + '</dd>' if model_mix else ''}"
            "</dl>"
            f"{'<p class=error>' + _h(item.get('error')) + '</p>' if item.get('error') else ''}"
            f"{'<p class=muted>' + _h(detail) + '</p>' if detail and not item.get('error') else ''}"
            "</article>"
        )
    return "".join(rows)


def _options(values, *, selected: str | None = None, include_blank: bool = False) -> str:
    opts = ["<option value=''>Default</option>"] if include_blank else []
    for value in values:
        value = str(value)
        sel = " selected" if value == selected else ""
        opts.append(f"<option value='{_h(value)}'{sel}>{_h(value)}</option>")
    return "".join(opts)


def _format_options(formats: dict[str, dict[str, Any]], *, selected: str | None = None, include_blank: bool = False) -> str:
    opts = ["<option value=''>Default</option>"] if include_blank else []
    for key, spec in formats.items():
        if spec.get("ui_hidden"):
            continue
        sel = " selected" if key == selected else ""
        label = spec.get("label", key)
        opts.append(f"<option value='{_h(key)}'{sel}>{_h(label)}</option>")
    return "".join(opts)


def _language_toggle(lang: str = "en") -> str:
    buttons = "".join(
        f"<button type='button' data-lang='{code}'"
        f"{' class=active' if code == lang else ''}"
        f" aria-pressed='{'true' if code == lang else 'false'}'>{label}</button>"
        for code, label in (("en", "EN"), ("zh", "中文"))
    )
    return (
        "<div class='lang-toggle' data-lang-toggle aria-label='Studio language'>"
        f"{buttons}</div>"
    )


def render_project(project: Project, notice: str = "", stage: str | None = None, *, lang: str = "en") -> str:
    selected = resolve_stage(project, stage)
    frames = []
    for index, stg in enumerate(project.stages, 1):
        status = project.stage_status(stg)
        classes = ["frame"]  # keep the existing class so the sprocket CSS still applies
        if stg == project.current_stage:
            classes.append("active")
        if str(status).lower() in _DONE_STATES:
            classes.append("done")
        elif stg != project.current_stage:
            classes.append("latent")
        if stg == selected:
            classes.append("selected")
        href = f"/projects/{_u(project.project_id)}?stage={_u(stg)}"
        frames.append(
            f"<a class='{' '.join(classes)}' href='{href}'>"
            f"<span class='no'>{index:02d}</span>"
            f"<span class='nm'>{_h(stg)}</span><span class='st'>{_h(t(status, lang))}</span></a>"
        )
    filmstrip = "<nav class='filmstrip' aria-label='Pipeline stages'>" + "".join(frames) + "</nav>"
    cost_warning = (
        f"<p class='muted cost-warning'>{t('Historical cost entries may be incomplete.', lang)}</p>"
        if project.has_incomplete_cost_history()
        else ""
    )
    stats = (
        "<div class='stats'>"
        f"<div class='stat'><span class='k'>{t('Status', lang)}</span><span class='v'>{_h(project.status)}</span></div>"
        f"<div class='stat'><span class='k'>{t('Gate', lang)}</span><span class='v'>{_h(project.current_stage or 'done')}</span></div>"
        f"<div class='stat'><span class='k'>{t('Estimated cost', lang)}</span><span class='v'>${project.total_cost():.2f}</span></div>"
        "</div>"
        f"{cost_warning}"
    )

    controls = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/resume' "
        "data-busy-message='Running next stage...'>"
        f"<button type='submit'>{t('Resume', lang)}</button></form>"
        f"<form method='post' action='/projects/{_u(project.project_id)}/resume-auto' "
        "data-busy-message='Running pipeline...'>"
        f"<button type='submit'>{t('Resume Auto', lang)}</button></form>"
    )

    artifact_rows = []
    for artifact in project_artifacts(project):
        href = f"/projects/{_u(project.project_id)}/artifact?path={_u(artifact['path'])}"
        artifact_rows.append(
            "<tr>"
            f"<td><a class='mono' href='{href}'>{_h(artifact['path'])}</a></td>"
            f"<td>{_h(artifact['kind'])}</td>"
            f"<td class='mono'>{artifact['size']}</td>"
            "</tr>"
        )

    artifacts_table = (
        "<table><thead><tr><th>Path</th><th>Kind</th><th>Bytes</th></tr></thead>"
        f"<tbody>{''.join(artifact_rows)}</tbody></table>"
    )
    body = (
        f"<script>window.STUDIO_PROJECT_STATE_URL='/projects/{_u(project.project_id)}/state.json';"
        f"window.STUDIO_PROJECT_VERSION='{_h(_project_version(project))}';</script>"
        "<header class='toolbar'><a class='back' href='/'>‹ Projects</a>"
        f"<span class='eyebrow'>{t('Project', lang)}</span>"
        f"<span class='crumb'>{_h(project.project_id)}</span>"
        f"<span class='sub'>{_h(project.idea)}</span>{_language_toggle(lang)}</header>"
        f"{'<p class=notice>' + _h(notice) + '</p>' if notice else ''}"
        "<main class='stack'>"
        "<section class='console'>"
        f"{filmstrip}{stats}<div class='controls'>{controls}</div></section>"
        + render_stage_workbench(project, selected, lang=lang) +
        f"<details class='all-files'><summary>{t('All files', lang)}</summary>"
        f"{artifacts_table}</details>"
        "</main>"
    )
    return page(project.project_id, body, lang=lang)


def render_stage_workbench(project: Project, stage: str | None = None, *, lang: str = "en") -> str:
    """Focused review surface for the selected pipeline stage."""
    ctx = workbench_context(project, stage)
    badge = f"<span class='badge {_status_class(ctx['status'])}'>{_h(t(ctx['status'], lang))}</span>"
    loading = _render_workbench_loading(ctx, lang=lang)
    body = _render_workbench_payload(project, ctx, lang=lang)
    prompt_review = (ctx.get("payload") or {}).get("kind") in {
        "keyframe_prompt_review", "video_prompt_review"
    }
    is_bible = str(ctx.get("stage")) == "bible" or str(ctx.get("selected")) == "bible"
    side = ""
    if not prompt_review:
        side = _render_workbench_decision(project, ctx, lang=lang)
        side += _render_knowledge_packets(project, ctx, lang=lang)
        side += _render_workbench_approval(project, ctx, lang=lang)
        if not is_bible:
            side += _render_workbench_revision(project, ctx, lang=lang)
            side += _render_workbench_actions(project, ctx, lang=lang)
        side += _render_adjustment_history(project)
    loading_cls = " is-loading" if loading else ""
    return (
        f"<section id='workbench' class='wide stage-workbench{loading_cls}'>"
        "<div class='section-head'><div>"
        f"<h2>{t('Stage Workbench', lang)}</h2><h3>{_h(t(ctx['stage_label'], lang))}</h3>"
        f"<p class='muted'>{_h(t(ctx['summary'], lang))}</p></div>{badge}</div>"
        f"{loading}"
        + (
            f"<div class='prompt-workbench-wrap'>{body}</div></section>"
            if prompt_review else
            f"<div class='workbench-layout'><div class='workbench-main'>{body}</div>"
            f"<aside class='workbench-side'>{side}</aside></div></section>"
        )
    )


def _render_workbench_decision(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    decision = ctx.get("decision") or {}
    if not decision:
        return ""
    stage = str(decision.get("stage") or ctx.get("stage") or "")
    default = str(decision.get("default") or "")
    choices = []
    for item in decision.get("choices") or []:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "")
        if not value:
            continue
        label = str(item.get("label") or value)
        button = t("Use the director", lang) if value == default else _h(label)
        description = str(item.get("description") or "")
        if value == default and not description:
            description = label
        choices.append(
            f"<form method='post' action='/projects/{_u(project.project_id)}/answer-decision' "
            "data-busy-message='Running next stage...'>"
            f"<input type='hidden' name='stage' value='{_h(stage)}'>"
            f"<input type='hidden' name='choice' value='{_h(value)}'>"
            f"<button type='submit' class='secondary'>{button}</button>"
            f"{'<p class=muted>' + _h(description) + '</p>' if description else ''}"
            "</form>"
        )
    return (
        f"<div class='workbench-box decision-card'><h3>{t('Creative decision', lang)}</h3>"
        f"<p>{_h(decision.get('question') or '')}</p>"
        f"{'<p class=muted>' + _h(decision.get('why_it_matters')) + '</p>' if decision.get('why_it_matters') else ''}"
        + "".join(choices)
        + "</div>"
    )


def _render_knowledge_packets(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    packets = ctx.get("knowledge_packets") or []
    if not packets:
        return ""
    cards = []
    for packet in packets:
        entries = "".join(
            "<li>"
            f"<b>{_h(entry.get('title'))}</b>"
            f"{' — ' + _h(entry.get('rationale')) if entry.get('rationale') else ''}"
            f"{'<br><span class=mono>' + _h(entry.get('source_path')) + '</span>' if entry.get('source_path') else ''}"
            "</li>"
            for entry in packet.get("entries") or []
        )
        cards.append(
            "<div class='knowledge-packet'>"
            f"<p><b>{_h(packet.get('purpose'))}: {_h(packet.get('target'))}</b></p>"
            f"<ul>{entries or '<li>No matching entries</li>'}</ul>"
            f"<form method='post' action='/projects/{_u(project.project_id)}/refresh-knowledge' "
            "data-busy-message='Refreshing knowledge...'>"
            f"<input type='hidden' name='purpose' value='{_h(packet.get('purpose'))}'>"
            f"<input type='hidden' name='target' value='{_h(packet.get('target'))}'>"
            f"<button type='submit' class='secondary'>{t('Refresh knowledge', lang)}</button></form></div>"
        )
    return f"<div class='workbench-box'><h3>{t('Filmmaking knowledge', lang)}</h3>" + "".join(cards) + "</div>"


def _render_workbench_loading(ctx: dict[str, Any], *, lang: str = "en") -> str:
    if str(ctx.get("status")).lower() != "running":
        return ""
    return (
        "<div class='stage-loading' role='status' aria-live='polite'>"
        "<span class='stage-spinner'></span>"
        f"<div><b>{t('Generating current stage...', lang)}</b>"
        f"<p class='muted'>{t('This page will refresh when new files are ready.', lang)}</p></div>"
        "</div>"
    )


def _render_workbench_approval(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    if ctx["stage"] == "done":
        return f"<div class='workbench-box'><h3>{t('Approval', lang)}</h3><p class='muted'>{t('All stages are approved.', lang)}</p></div>"
    payload = ctx.get("payload") or {}
    if payload.get("kind") in {"keyframe_prompt_review", "video_prompt_review"}:
        kind = payload.get("gate")
        label = (
            "Confirm all & generate keyframes"
            if kind == "keyframes" else "Confirm all & generate videos"
        )
        disabled = "" if ctx["can_approve"] and payload.get("batch_ready") else " disabled"
        return (
            f"<form method='post' action='/projects/{_u(project.project_id)}/confirm-prompts'>"
            f"<input type='hidden' name='kind' value='{_h(kind)}'>"
            f"<button class='primary' type='submit'{disabled}>{t(label, lang)}</button></form>"
        )
    disabled = "" if ctx["can_approve"] else " disabled"
    hint = "" if ctx["can_approve"] else f"<p class='muted'>{t('The current stage must finish before approval.', lang)}</p>"
    return (
        f"<div class='workbench-box'><h3>{t('Approval', lang)}</h3>"
        f"<form method='post' action='/projects/{_u(project.project_id)}/approve' "
        "data-busy-message='Approving stage...'>"
        f"<button class='primary' type='submit'{disabled}>{t('Approve this stage', lang)}</button></form>{hint}</div>"
    )


def _render_workbench_revision(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    targets = ctx.get("revision_targets") or []
    if not targets:
        return (
            f"<div class='workbench-box'><h3>{t('Ask for changes', lang)}</h3>"
            f"<p class='muted'>{t('No editable artifact is available for this stage yet.', lang)}</p></div>"
        )
    default = (ctx.get("revision_target") or {}).get("path")
    options = "".join(
        f"<option value='{_h(target['path'])}'{' selected' if target['path'] == default else ''}>"
        f"{_h(target['label'])}</option>"
        for target in targets
    )
    placeholder_revision = t(
        "Make the story darker, make this character more specific, or simplify the shot list.",
        lang,
    )
    return (
        f"<div class='workbench-box'><h3>{t('Ask for changes', lang)}</h3>"
        f"<form method='post' action='/projects/{_u(project.project_id)}/adjust-artifact' "
        "data-busy-message='Revising with LLM...'>"
        f"<label>{t('Artifact', lang)}<select name='path'>{options}</select></label>"
        f"<label>{t('Revision request', lang)}"
        "<textarea name='instruction' "
        f"placeholder='{_h(placeholder_revision)}'></textarea>"
        f"</label><button type='submit'>{t('Revise current stage', lang)}</button></form></div>"
    )


def _render_workbench_actions(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    actions = ctx.get("asset_actions") or []
    forms = []
    for action in actions:
        forms.append(
            f"<form method='post' action='/projects/{_u(project.project_id)}/regenerate-asset' "
            f"data-busy-message='{_h(action.get('label') or 'Regenerating asset')}...'>"
            f"<input type='hidden' name='kind' value='{_h(action.get('kind'))}'>"
            f"<input type='hidden' name='shot_id' value='{_h(action.get('shot_id'))}'>"
            f"<input type='hidden' name='rel_path' value='{_h(action.get('rel_path'))}'>"
            f"<button type='submit' class='secondary'>{_h(action.get('label'))}</button></form>"
        )
    if ctx.get("selected") == "video":
        shot_placeholder = t("shot id, e.g. sh-001", lang)
        forms.append(
            f"<form method='post' action='/projects/{_u(project.project_id)}/regenerate' "
            "class='inline' data-busy-message='Regenerating shot...'>"
            f"<input name='shot' placeholder='{_h(shot_placeholder)}'>"
            f"<button type='submit'>{t('Regenerate shot', lang)}</button></form>"
        )
    if not forms:
        return (
            f"<div class='workbench-box'><h3>{t('Regenerate', lang)}</h3>"
            f"<p class='muted'>{t('No generated assets are available for this stage.', lang)}</p></div>"
        )
    return (
        f"<div class='workbench-box'><h3>{t('Regenerate', lang)}</h3>"
        f"<div class='asset-actions'>{''.join(forms)}</div></div>"
    )


def _render_workbench_payload(project: Project, ctx: dict[str, Any], *, lang: str = "en") -> str:
    payload = ctx.get("payload") or {}
    kind = payload.get("kind")
    if kind == "style":
        return _render_workbench_style(project, payload, lang=lang)
    if kind == "plot":
        return _render_workbench_plot(project, payload.get("plot") or {}, lang=lang)
    if kind == "script":
        return _render_workbench_script(project, payload.get("scenes") or [], lang=lang)
    if kind == "bible":
        return _render_workbench_bible(project, payload, lang=lang) + _render_references_view(
            project, stage=ctx.get("stage"), lang=lang
        )
    if kind in {"keyframe_prompt_review", "video_prompt_review"}:
        return _render_prompt_review(project, payload, can_confirm=ctx.get("can_approve", False), lang=lang)
    if kind == "keyframe_review":
        return _render_workbench_shots(project, payload.get("shots") or [], "keyframes", lang=lang)
    if kind == "video_review":
        return _render_workbench_manual_review(project, lang=lang)
    if kind == "audio":
        return _render_workbench_audio(project, payload, lang=lang)
    if kind == "assemble":
        return _render_workbench_assemble(project, payload, lang=lang)
    return f"<p class='muted'>{t('Project complete.', lang)}</p>"


def _add_shot_form(project: Project, after_shot_id: str, *, lang: str = "en") -> str:
    """A collapsed form that drafts and inserts a new shot after ``after_shot_id``
    (blank = at the very start). Available at every shot-editing gate."""
    label = t("Add shot after this one", lang) if after_shot_id else t("Add shot at start", lang)
    placeholder = t("Describe what the new shot should show.", lang)
    return (
        f"<details class='add-shot'><summary>{_h(label)}</summary>"
        f"<form method='post' action='/projects/{_u(project.project_id)}/add-shot' "
        "data-busy-message='Drafting the new shot...'>"
        f"<input type='hidden' name='after_shot_id' value='{_h(after_shot_id)}'>"
        f"<label>{t('Describe the new shot', lang)}"
        f"<textarea name='description' required placeholder='{_h(placeholder)}'></textarea></label>"
        f"<button type='submit' class='secondary'>{t('Add shot', lang)}</button>"
        "</form></details>"
    )


def _delete_shot_form(project: Project, shot_id: str, *, stage: str, lang: str = "en") -> str:
    return (
        f"<form class='clip-delete' method='post' "
        f"action='/projects/{_u(project.project_id)}/delete-clip' "
        f"onsubmit=\"return confirm('{_h(t('Delete this clip?', lang))}');\">"
        f"<input type='hidden' name='shot_id' value='{_h(shot_id)}'>"
        f"<input type='hidden' name='stage' value='{_h(stage)}'>"
        f"<button type='submit' class='danger'>{_h(t('Delete shot', lang))}</button>"
        "</form>"
    )


def _clip_duration_form(
    project: Project,
    shot: dict[str, Any],
    *,
    capabilities,
    lang: str = "en",
) -> str:
    """Per-clip duration editor; blank = the video model's default length. Bounds come
    from the configured video provider's capabilities, not hardcoded numbers."""
    duration = shot.get("duration_s")
    value = f" value='{_h(duration)}'" if duration is not None else ""
    return (
        f"<form class='clip-duration' method='post' "
        f"action='/projects/{_u(project.project_id)}/set-clip-duration'>"
        f"<input type='hidden' name='shot_id' value='{_h(shot.get('id'))}'>"
        f"<label>{_h(t('Duration (s)', lang))} "
        f"<input type='number' name='seconds' min='{_h(capabilities.min_duration_s)}' "
        f"max='{_h(capabilities.max_duration_s)}' placeholder='{_h(t('Model default', lang))}'"
        f"{value}></label>"
        f"<button type='submit' class='secondary'>{_h(t('Update', lang))}</button>"
        "</form>"
    )


def _render_prompt_review(
    project: Project,
    payload: dict[str, Any],
    *,
    can_confirm: bool,
    lang: str = "en",
) -> str:
    shots = payload.get("shots") or []
    if not shots:
        return f"<p class='muted'>{t('No shots yet.', lang)}</p>"
    gate = str(payload.get("gate") or "")
    selected = shots[0]
    shot_options = "".join(
        f"<option value='{_h(shot.get('id'))}'>{_h(shot.get('id'))} · "
        f"{_h(shot.get('prompt_status'))} · {shot.get('revision_count', 0)} rev</option>"
        for shot in shots
    )
    rail = "".join(
        f"<a href='#prompt-{_h(shot.get('id'))}' class='prompt-shot { _h(shot.get('prompt_status')) }'>"
        f"<b>{_h(shot.get('id'))}</b><span>{_h(shot.get('prompt_status'))}</span>"
        f"<small>{shot.get('revision_count', 0)} revision(s)</small></a>"
        for shot in shots
    )
    confirm_label = (
        "Confirm all & generate keyframes"
        if gate == "keyframes" else "Confirm all & generate videos"
    )
    disabled = "" if can_confirm and payload.get("batch_ready") else " disabled"
    confirm = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/confirm-prompts' "
        "data-busy-message='Running approved prompts...'>"
        f"<input type='hidden' name='kind' value='{_h(gate)}'>"
        f"<button class='primary' type='submit'{disabled}>{t(confirm_label, lang)}</button></form>"
    )
    # A disabled confirm must say WHY. When the stage is approvable but the batch isn't ready,
    # the blockers are the shots whose prompts fail validation — list them (with jump links) next
    # to the button so an off-screen invalid shot can't leave the user staring at a dead control.
    blockers = ""
    if can_confirm and not payload.get("batch_ready"):
        invalid = [shot for shot in shots if shot.get("validation_errors")]
        if invalid:
            items = "".join(
                f"<li><a href='#prompt-{_h(shot.get('id'))}'>{_h(shot.get('id'))}</a>: "
                f"{_h('; '.join(shot.get('validation_errors') or []))}</li>"
                for shot in invalid
            )
            blockers = (
                f"<div class='prompt-blockers'><b>{t('Fix these shots before confirming', lang)}</b>"
                f"<ul>{items}</ul></div>"
            )
    errors = "".join(
        f"<li>{_h(error)}</li>" for error in selected.get("validation_errors") or []
    )
    status = (
        f"<div class='prompt-errors'><b>{t('Prompt invalid', lang)}</b><ul>{errors}</ul></div>"
        if errors else ""
    )
    prompt_label = (
        "Motion-grid prompt" if selected.get("prompt_kind") == "grid"
        else "Static prompt" if gate == "keyframes" else "Motion prompt"
    )
    keyframe = ""
    if selected.get("keyframe_rel"):
        media = f"/projects/{_u(project.project_id)}/media?path={_u(selected['keyframe_rel'])}"
        keyframe = f"<img class='prompt-keyframe' src='{media}' alt='{_h(selected.get('id'))} keyframe'>"
    panel_plan = ""
    if selected.get("prompt_kind") == "grid":
        panel_plan = "<ol class='panel-plan'>" + "".join(
            f"<li>{_h(beat)}</li>" for beat in selected.get("panel_plan") or []
        ) + "</ol>"
    later = ""
    if gate == "keyframes":
        later = (
            f"<details><summary>{t('Later video intent', lang)} — "
            f"{t('Not sent to image model', lang)}</summary>"
            f"<pre>{_h(json.dumps(selected.get('later_video_intent') or {}, ensure_ascii=False, indent=2))}</pre>"
            "</details>"
        )
    references = "".join(
        f"<li>{_h(ref)}</li>" for ref in selected.get("references") or []
    )
    history_rel = "storyboard/conversations/prompt-review.jsonl"
    history_link = (
        _artifact_link(project, history_rel, t("Conversation history", lang))
        if project.path(*history_rel.split("/")).is_file() else ""
    )
    records = payload.get("conversation") or []
    conversation = "".join(
        f"<div class='conversation-turn {_h(record.get('role'))}'><b>{_h(record.get('role'))}</b>"
        f"<p>{_h(record.get('text'))}</p></div>"
        for record in records[-12:]
    ) or f"<p class='muted'>{t('No prompt revisions yet.', lang)}</p>"
    last_mode = next(
        (str(record.get("vision_mode") or "") for record in reversed(records) if record.get("role") == "assistant"),
        "",
    )
    disclosure = (
        t("Images directly inspected", lang) if last_mode == "vision"
        else t("Text-only revision", lang) if last_mode else ""
    )
    chat = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/prompt-chat'>"
        f"<input type='hidden' name='gate' value='{_h(gate)}'>"
        f"<label>{t('Selected shot', lang)}<select name='shot_id'>{shot_options}</select></label>"
        f"<label class='check'><input type='checkbox' name='apply_to_all' value='1'>"
        f"{t('Apply to all shots', lang)}</label>"
        f"<label>{t('Project-aware conversation', lang)}"
        "<textarea name='message' required placeholder='Describe the exact change; keep talking until it is right.'></textarea></label>"
        f"<button type='submit'>{t('Revise prompt', lang)}</button></form>"
    )
    # Structural shot edits (add / delete / clip duration) live at both prompt gates so
    # the shot list can change wherever a human is reviewing it. Duration edits appear
    # only at the clip-mode keyframe gate — the natural place to size each clip.
    clip_durations = gate == "keyframes" and "clip" in project.stages
    capabilities = project_video_capabilities(project) if clip_durations else None

    def _structure_controls(shot: dict[str, Any]) -> str:
        sid = str(shot.get("id") or "")
        forms = []
        if clip_durations and capabilities is not None:
            forms.append(
                _clip_duration_form(project, shot, capabilities=capabilities, lang=lang)
            )
        if len(shots) > 1:
            forms.append(_delete_shot_form(project, sid, stage=gate, lang=lang))
        forms.append(_add_shot_form(project, sid, lang=lang))
        return "<div class='shot-structure'>" + "".join(forms) + "</div>"

    remaining_prompts = "".join(
        "<article class='additional-prompt' "
        f"id='prompt-{_h(shot.get('id'))}'><header><h3>{_h(shot.get('id'))} · "
        f"{_h(shot.get('prompt_kind'))}</h3>"
        f"{_artifact_link(project, shot.get('prompt_rel'), t('Open raw file', lang))}</header>"
        + (
            "<div class='prompt-errors'><b>"
            + t("Prompt invalid", lang)
            + "</b><ul>"
            + "".join(f"<li>{_h(error)}</li>" for error in shot.get("validation_errors") or [])
            + "</ul></div>"
            if shot.get("validation_errors") else ""
        )
        + f"<pre class='exact-prompt'>{_h(shot.get('prompt_text'))}</pre>"
        + _structure_controls(shot)
        + "</article>"
        for shot in shots[1:]
    )
    add_at_start = _add_shot_form(project, "", lang=lang)
    return (
        "<div class='prompt-review-layout'>"
        f"<aside class='prompt-shot-rail'><h3>{t('Shots', lang)}</h3>{rail}{confirm}{blockers}"
        f"{add_at_start}"
        f"<p class='muted'>{t('Cost', lang)}: ${payload.get('cost', 0):.2f} / "
        f"{payload.get('cost_cap') if payload.get('cost_cap') is not None else '∞'}</p></aside>"
        f"<main class='prompt-review-main' id='prompt-{_h(selected.get('id'))}'>{keyframe}"
        f"<header><h3>{t(prompt_label, lang)} · {_h(selected.get('id'))}</h3>"
        f"{_artifact_link(project, selected.get('prompt_rel'), t('Open raw file', lang))}</header>"
        f"{status}{panel_plan}<pre class='exact-prompt'>{_h(selected.get('prompt_text'))}</pre>"
        f"{later}<h4>{t('References', lang)}</h4><ul>{references or '<li class=muted>none</li>'}</ul>"
        f"{_structure_controls(selected)}"
        f"{remaining_prompts}</main>"
        f"<aside class='prompt-conversation'><h3>{t('Project-aware conversation', lang)}</h3>"
        f"<p class='muted'>{_h(disclosure)}</p>{conversation}{chat}{history_link}</aside></div>"
    )


def _render_workbench_manual_review(project: Project, *, lang: str = "en") -> str:
    reviews = shot_reviews(project)
    if not reviews:
        return f"<p class='muted'>{t('No shots yet.', lang)}</p>"
    cards = "".join(_render_shot_card(project, r, lang=lang) for r in reviews)
    return f"<div class='shots-grid'>{cards}</div>"
def _render_workbench_style(project: Project, payload: dict[str, Any], *, lang: str = "en") -> str:
    media = f"/projects/{_u(project.project_id)}/media?path={_u(payload.get('sample_rel') or '')}"
    image = (
        f"<figure class='bible-image'><img src='{media}' alt='style sample'>"
        f"<figcaption>{t('Style sample', lang)}</figcaption></figure>"
        if payload.get("sample_rel") else ""
    )
    placeholder = t("Make it warmer, less cartoonish, more like 90s film stock.", lang)
    feedback_form = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/style-feedback' "
        "data-busy-message='Refining style...'>"
        f"<label>{t('Adjust the style', lang)}"
        f"<textarea name='feedback' placeholder='{_h(placeholder)}'></textarea></label>"
        f"<button type='submit'>{t('Refine style', lang)}</button></form>"
    )
    return (
        "<article class='workbench-readable'>"
        f"<h3>{_h(payload.get('label') or t('Visual Style', lang))}</h3>"
        f"<div class='workbench-media-row'>{image}</div>"
        f"<pre class='style-md'>{_h(payload.get('style_md') or '')}</pre>"
        f"{feedback_form}</article>"
    )


def _render_workbench_plot(project: Project, plot: dict[str, Any], *, lang: str = "en") -> str:
    if not plot:
        return f"<p class='muted'>{t('No plot yet.', lang)}</p>"
    chars = "".join(
        f"<li><b>{_h(c.get('name'))}</b> {_h(c.get('role') or '')}: {_h(c.get('description') or '')}</li>"
        for c in plot.get("characters", []) if isinstance(c, dict)
    )
    themes = ", ".join(str(t_) for t_ in plot.get("themes", []))
    return (
        "<article class='workbench-readable'>"
        f"<h3>{_h(plot.get('logline') or t('Plot', lang))}</h3>"
        f"<p>{_h(plot.get('synopsis') or '')}</p>"
        f"<p><b>{t('Themes', lang)}</b>: {_h(themes)}</p>"
        f"<h4>{t('Characters', lang)}</h4><ul>{chars or '<li class=muted>' + t('No characters listed.', lang) + '</li>'}</ul>"
        f"<p class='muted'>{t('Raw', lang)}: {_artifact_link(project, 'story/plot.json')}</p></article>"
    )


def _render_workbench_script(project: Project, scenes: list[dict[str, Any]], *, lang: str = "en") -> str:
    if not scenes:
        return f"<p class='muted'>{t('No script yet.', lang)}</p>"
    cards = []
    for scene in scenes[:12]:
        beats = "".join(f"<li>{_h(beat)}</li>" for beat in scene.get("beats", []))
        dialogue = "".join(
            f"<li><b>{_h(line.get('character'))}</b>: {_h(line.get('line'))}</li>"
            for line in scene.get("dialogue", []) if isinstance(line, dict)
        )
        cards.append(
            "<article class='workbench-card'>"
            f"<header><b>{t('Scene', lang)} {_h(scene.get('scene'))}</b><span>{_h(scene.get('heading'))}</span></header>"
            f"<p>{_h(scene.get('objective') or scene.get('emotional_turn') or '')}</p>"
            f"<h4>{t('Beats', lang)}</h4><ul>{beats or '<li class=muted>' + t('No beats.', lang) + '</li>'}</ul>"
            f"<h4>{t('Dialogue', lang)}</h4><ul>{dialogue or '<li class=muted>' + t('No dialogue.', lang) + '</li>'}</ul></article>"
        )
    more = (
        f"<p class='muted'>{t('Showing first {shown} of {total} scenes.', lang).format(shown=12, total=len(scenes))}</p>"
        if len(scenes) > 12 else ""
    )
    return more + f"<div class='workbench-grid'>{''.join(cards)}</div><p class='muted'>{t('Raw', lang)}: {_artifact_link(project, 'story/script.json')}</p>"


_BIBLE_IMAGE_LABELS = {
    "reference.png": "Reference",
    "Identity reference": "Identity reference",
}


_HEX_RE = re.compile(r"^#(?:[0-9A-Fa-f]{3,4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")


def _palette_chips(value: Any) -> str:
    """Render a palette (list of hex strings, or a string/dict) as color chips.

    Falls back to humanized text when the value is not a plain color list, so we never
    dump a raw Python list/dict repr into the page.
    """
    colors = value if isinstance(value, list) else ([value] if isinstance(value, str) and value.strip().startswith("#") else [])
    chips = []
    for raw in colors:
        token = str(raw).strip()
        if not token:
            continue
        style = f" style='background:{token}'" if _HEX_RE.match(token) else ""
        chips.append(f"<span class='swatch'{style}></span><code>{_h(token)}</code>")
    if chips:
        return f"<div class='palette-chips'>{''.join(chips)}</div>"
    text = humanize_board_value(value)
    return f"<span>{_h(text)}</span>" if text else ""


def _point_list(label: str, value: Any, *, lang: str = "en") -> str:
    """Render a list field as a labeled point-form block; humanize non-list scalars."""
    if isinstance(value, list):
        items = "".join(
            f"<li>{_h(humanize_board_value(item))}</li>"
            for item in value
            if str(item).strip()
        )
        if not items:
            return ""
        return f"<div class='id-points'><h5>{_h(t(label, lang))}</h5><ul>{items}</ul></div>"
    text = humanize_board_value(value)
    if not text:
        return ""
    return f"<p><b>{_h(t(label, lang))}</b>: {_h(text)}</p>"


def _location_card_body(data: dict[str, Any], *, lang: str = "en") -> str:
    """Render a location entry as clear point form: overview, palette chips, lighting,
    materials, hero props, and continuity rules."""
    parts = []
    description = humanize_board_value(data.get("description"))
    if description:
        parts.append(f"<p>{_h(description)}</p>")
    palette = _palette_chips(data.get("palette"))
    if palette:
        parts.append(f"<p class='palette-row'><b>{_h(t('Palette', lang))}</b>: {palette}</p>")
    parts.append(_point_list("Lighting", data.get("lighting"), lang=lang))
    parts.append(_point_list("Materials", data.get("materials"), lang=lang))
    parts.append(_point_list("Hero props", data.get("hero_props"), lang=lang))
    parts.append(_point_list("Continuity rules", data.get("continuity_rules"), lang=lang))
    return "".join(part for part in parts if part)


def _bible_image_figure(src: str, label: str, *, lang: str = "en") -> str:
    if label.startswith("Appearance state reference: "):
        state = label.split(": ", 1)[1]
        visible = f"{t('Appearance state reference', lang)}: {state}"
    else:
        visible = t(_BIBLE_IMAGE_LABELS.get(label, label), lang)
    return (
        "<figure class='bible-image'>"
        f"<img src='{src}' alt='{_h(visible)}'>"
        f"<figcaption>{_h(visible)}</figcaption></figure>"
    )


def _bible_card_controls(project: Project, kind: str, slug: str, *, text_ahead: bool, lang: str = "en") -> str:
    pid = _u(project.project_id)
    ref_rel = f"bible/{'characters' if kind == 'character' else 'locations'}/{slug}/reference.png"
    comment_ph = t("Describe the change in plain words, e.g. make the wardrobe warmer.", lang)
    text_form = (
        f"<form class='bible-controls' method='post' action='/projects/{pid}/regenerate-bible-text' "
        "data-busy-message='Regenerating text...'>"
        f"<input type='hidden' name='kind' value='{_h(kind)}'>"
        f"<input type='hidden' name='slug' value='{_h(slug)}'>"
        f"<label>{t('Comment', lang)}"
        f"<textarea name='instruction' required placeholder='{_h(comment_ph)}'></textarea></label>"
        f"<button type='submit' class='secondary'>{t('Regenerate text', lang)}</button></form>"
    )
    if text_ahead:
        img_label = t("Approve text & regenerate image", lang)
        img_cls = "primary"
        banner = f"<p class='bible-banner'>⚠ {t('Text changed since last image', lang)}</p>"
    else:
        img_label = t("Regenerate image", lang)
        img_cls = "secondary"
        banner = ""
    img_form = (
        f"<form class='bible-controls' method='post' action='/projects/{pid}/regenerate-asset' "
        f"data-busy-message='{_h(img_label)}...'>"
        "<input type='hidden' name='kind' value='bible_asset'>"
        "<input type='hidden' name='shot_id' value=''>"
        f"<input type='hidden' name='rel_path' value='{_h(ref_rel)}'>"
        f"<button type='submit' class='{img_cls}'>{_h(img_label)}</button></form>"
    )
    return f"<div class='bible-card-controls'>{banner}{text_form}{img_form}</div>"


def _description_rows_html(value: Any) -> str:
    """Render a character/location description (structured dict or free text) as labeled
    rows / a paragraph — never a raw dict repr. Surfaces the current source-of-truth text
    (character.json / location.json) so a revision is always visible in the detail popup.
    """
    as_dict = coerce_board_dict(value)
    if as_dict is not None:
        rows = []
        for key, sub in as_dict.items():
            text = humanize_board_value(sub)
            if text:
                label = key.replace("_", " ").strip().capitalize()
                rows.append(f"<div class='id-row'><dt>{_h(label)}</dt><dd>{_h(text)}</dd></div>")
        return f"<dl class='identity-board'>{''.join(rows)}</dl>" if rows else ""
    text = humanize_board_value(value)
    return f"<p>{_h(text)}</p>" if text else ""


def _character_detail_body(data: dict[str, Any], board: dict[str, Any], states: dict[str, Any], *, lang: str = "en") -> str:
    """Read-only detail block for a character's popup: description, identity board, states."""
    desc = _description_rows_html(data.get("description") or data.get("visual_description"))
    desc_section = f"<section><h4>{_h(t('Description', lang))}</h4>{desc}</section>" if desc else ""
    board_section = _identity_board_html(data, board)
    state_items = "".join(
        "<li>"
        f"<b>{_h(state.get('label') or state.get('id'))}</b> "
        f"({_h(state.get('kind') or '')}) — {_h(state.get('description') or '')}"
        "</li>"
        for state in (states.get("states") or [])
        if isinstance(state, dict)
    )
    states_section = (
        f"<section><h4>{_h(t('Visual states', lang))}</h4><ul>{state_items}</ul></section>"
        if state_items else ""
    )
    return desc_section + board_section + states_section


def _location_detail_body(data: dict[str, Any], *, lang: str = "en") -> str:
    """Read-only detail block for a location's popup."""
    return _location_card_body(data, lang=lang)


def _bible_detail_dialog(
    project: Project, kind: str, slug: str, name: str, body_html: str, *, lang: str = "en"
) -> tuple[str, str]:
    """Return (trigger_html, dialog_html) for a read-only detail popup.

    Rendered from the current on-disk payload, so the existing state.json poller's
    auto-reload makes the popup reflect revised text after a regeneration.
    """
    pid = _u(project.project_id)
    folder = "characters" if kind == "character" else "locations"
    filename = "character.json" if kind == "character" else "location.json"
    raw_rel = f"bible/{folder}/{slug}/{filename}"
    raw_url = f"/projects/{pid}/artifact?path={_u(raw_rel)}"
    dlg_id = f"bible-detail-{kind}-{slug}"
    title_id = f"{dlg_id}-title"
    trigger = (
        f"<button type='button' class='bible-details-link' "
        f"data-bible-detail-open='{_h(dlg_id)}' aria-haspopup='dialog'>"
        f"🔍 {_h(t('View details', lang))}</button>"
    )
    dialog = (
        f"<dialog class='bible-detail-dialog' id='{_h(dlg_id)}' aria-labelledby='{_h(title_id)}'>"
        "<form method='dialog' class='bible-detail-head'>"
        f"<h3 id='{_h(title_id)}'>{_h(name)}</h3>"
        f"<button value='close' class='bible-detail-close' aria-label='{_h(t('Close', lang))}'>×</button>"
        "</form>"
        f"<div class='bible-detail-body'>{body_html or '<p class=muted>' + _h(t('No details yet.', lang)) + '</p>'}</div>"
        f"<div class='bible-detail-foot'><a href='{raw_url}'>{_h(t('Open raw file', lang))} ({_h(filename)})</a></div>"
        "</dialog>"
    )
    return trigger, dialog


def _render_workbench_bible(project: Project, payload: dict[str, Any], *, lang: str = "en") -> str:
    media = lambda rel: f"/projects/{_u(project.project_id)}/media?path={_u(rel)}"
    cards = []
    for character in payload.get("characters") or []:
        images = "".join(
            _bible_image_figure(media(img["rel"]), img["label"], lang=lang)
            for img in character.get("images", [])
        )
        data = character.get("character") or {}
        board = character.get("identity_board") or {}
        states = character.get("states") or {}
        slug = character.get("slug")
        name = character.get("name")
        body = _character_detail_body(data, board, states, lang=lang)
        trigger, dialog = _bible_detail_dialog(project, "character", slug, name, body, lang=lang)
        cards.append(
            "<article class='workbench-card bible-card'>"
            f"<header><b>{_h(name)}</b><span>{t('character', lang)}</span></header>"
            f"<div class='bible-media-row'>{images}</div>"
            f"<div class='bible-card-meta'>{trigger}</div>"
            f"{_bible_card_controls(project, 'character', slug, text_ahead=bool(character.get('text_ahead')), lang=lang)}"
            f"{dialog}"
            "</article>"
        )
    for location in payload.get("locations") or []:
        images = "".join(
            _bible_image_figure(media(img["rel"]), img["label"], lang=lang)
            for img in location.get("images", [])
        )
        data = location.get("location") or {}
        slug = location.get("slug")
        name = location.get("name")
        body = _location_detail_body(data, lang=lang)
        trigger, dialog = _bible_detail_dialog(project, "location", slug, name, body, lang=lang)
        cards.append(
            "<article class='workbench-card bible-card'>"
            f"<header><b>{_h(name)}</b><span>{t('location', lang)}</span></header>"
            f"<div class='bible-media-row'>{images}</div>"
            f"<div class='bible-card-meta'>{trigger}</div>"
            f"{_bible_card_controls(project, 'location', slug, text_ahead=bool(location.get('text_ahead')), lang=lang)}"
            f"{dialog}"
            "</article>"
        )
    return f"<div class='workbench-grid bible-grid'>{''.join(cards) or '<p class=muted>' + t('No bible assets yet.', lang) + '</p>'}</div>"


def _render_workbench_shots(project: Project, shots: list[dict[str, Any]], kind: str, *, lang: str = "en") -> str:
    if not shots:
        return f"<p class='muted'>{t('No shots yet.', lang)}</p>"
    media = lambda rel: f"/projects/{_u(project.project_id)}/media?path={_u(rel)}"
    cards = []
    for shot in shots[:24]:
        sid = shot.get("id")
        media_html = ""
        if kind in {"clip", "video", "review"} and shot.get("clip_rel"):
            poster = f" poster='{media(shot['keyframe_rel'])}'" if shot.get("keyframe_rel") else ""
            media_html = f"<video controls preload='metadata'{poster} src='{media(shot['clip_rel'])}'></video>"
        elif shot.get("keyframe_rel"):
            media_html = f"<img src='{media(shot['keyframe_rel'])}' alt='{_h(sid)} keyframe'>"
        links = []
        for key, label in (
            ("keyframe_prompt_rel", "keyframe prompt"),
            ("video_prompt_rel", "video prompt"),
            ("audio_prompt_rel", "audio prompt"),
        ):
            if shot.get(key):
                links.append(_artifact_link(project, shot[key], t(label, lang)))
        feedback_form = ""
        if kind == "keyframes" and shot.get("keyframe_rel"):
            feedback_form = (
                f"<form class='keyframe-feedback' method='post' "
                f"action='/projects/{_u(project.project_id)}/keyframe-feedback' "
                "data-busy-message='Revising keyframe prompt...'>"
                f"<input type='hidden' name='shot_id' value='{_h(sid)}'>"
                f"<label>{t('Feedback', lang)}"
                "<textarea name='feedback' required "
                "placeholder='Make the face less photoreal, fix the coat color, or move the character left.'>"
                "</textarea></label>"
                f"<button type='submit' class='secondary'>{_h(t('Revise prompt & regenerate keyframe', lang))}</button>"
                "</form>"
            )
        clip_controls = ""
        if kind == "keyframes":
            clip_controls = "<div class='shot-structure'>" + (
                _delete_shot_form(project, sid, stage="keyframes", lang=lang)
                if len(shots) > 1 else ""
            ) + _add_shot_form(project, sid, lang=lang) + "</div>"
        duration_label = (
            f"{_h(shot.get('duration_s'))}s"
            if shot.get("duration_s") is not None else _h(t("Model default", lang))
        )
        cards.append(
            "<article class='workbench-card'>"
            f"{media_html}<header><b>{_h(sid)}</b><span>{t('scene', lang)} {_h(shot.get('scene'))} · {duration_label}</span></header>"
            f"<p><b>{t('Camera', lang)}</b>: {_h(shot.get('camera'))} {_h(shot.get('camera_movement'))}</p>"
            f"<p><b>{t('Action', lang)}</b>: {_h(shot.get('action'))}</p>"
            f"<p>{_h(shot.get('description') or '')}</p>"
            f"{'<p class=muted>' + ' · '.join(links) + '</p>' if links else ''}"
            f"{feedback_form}{clip_controls}</article>"
        )
    more = (
        f"<p class='muted'>{t('Showing first {shown} of {total} shots.', lang).format(shown=24, total=len(shots))}</p>"
        if len(shots) > 24 else ""
    )
    return more + f"<div class='workbench-grid'>{''.join(cards)}</div>"


def _render_workbench_audio(project: Project, payload: dict[str, Any], *, lang: str = "en") -> str:
    prompts = "".join(f"<li>{_artifact_link(project, rel)}</li>" for rel in payload.get("prompts") or [])
    audio = "".join(f"<li>{_artifact_link(project, rel)}</li>" for rel in payload.get("audio_files") or [])
    if payload.get("mode") != "native_video":
        return (
            f"<div class='workbench-readable'><h3>{t('Audio Assets', lang)}</h3>"
            f"<h4>{t('Prompts', lang)}</h4><ul>{prompts or '<li class=muted>' + t('No audio prompts yet.', lang) + '</li>'}</ul>"
            f"<h4>{t('Files', lang)}</h4><ul>{audio or '<li class=muted>' + t('No audio files yet.', lang) + '</li>'}</ul></div>"
        )
    sources = "".join(
        f"<li>{_artifact_link(project, rel)}</li>"
        for rel in payload.get("source_clips") or []
    )
    return (
        f"<div class='workbench-readable'><h3>{t('Audio Assets', lang)}</h3>"
        f"<h4>{t('Prompts', lang)}</h4><ul>{prompts or '<li class=muted>' + t('No prompts yet.', lang) + '</li>'}</ul>"
        f"<h4>{t('Extracted tracks', lang)}</h4><ul>{audio or '<li class=muted>' + t('No audio files yet.', lang) + '</li>'}</ul>"
        f"<h4>{t('Source clips', lang)}</h4><ul>{sources or '<li class=muted>' + t('No source clips.', lang) + '</li>'}</ul>"
        f"<p class='muted'>{t('Review dialogue, effects, ambience, and music absence manually.', lang)}</p></div>"
    )


def _render_workbench_assemble(project: Project, payload: dict[str, Any], *, lang: str = "en") -> str:
    preview = ""
    if payload.get("output"):
        media = f"/projects/{_u(project.project_id)}/media?path={_u(payload['output'])}"
        preview = f"<video controls src='{media}'></video>"
    return (
        f"<div class='workbench-readable'><h3>{t('Assembly', lang)}</h3>"
        + (preview or f"<p class='muted'>{t('No assembled output yet.', lang)}</p>")
        + f"<p class='muted'>{t('Timeline', lang)}: {_artifact_link(project, 'edit/timeline.json') if project.path('edit', 'timeline.json').is_file() else 'edit/timeline.json'}</p>"
        "</div>"
    )



def _artifact_link(project: Project, rel: str, label: str | None = None) -> str:
    return f"<a href='/projects/{_u(project.project_id)}/artifact?path={_u(rel)}'>{_h(label or rel)}</a>"


def _script_scenes(script: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = []
    for episode in script.get("episodes", []):
        scenes.extend(episode.get("scenes", []))
    return scenes


def _render_references_view(project: Project, *, stage: str | None = None, lang: str = "en") -> str:
    media = lambda rel: f"/projects/{_u(project.project_id)}/media?path={_u(rel)}"
    # Round-trip the viewed stage so the post-upload re-render stays on this workbench
    # instead of snapping to current_stage (where the references section is not rendered).
    stage_field = f"<input type='hidden' name='stage' value='{_h(stage or '')}'>"
    upload_options = "".join(
        f"<option value='{_h(value)}'>{_h(label)}</option>"
        for value, label in _reference_target_options(project, include_auto=True, lang=lang)
    )
    retarget_options = "".join(
        f"<option value='{_h(value)}'>{_h(label)}</option>"
        for value, label in _reference_target_options(project, lang=lang)
    )
    alias_start = next_reference_alias(project).removeprefix("@image")
    form = (
        f"<form method='post' action='/projects/{_u(project.project_id)}/upload-reference' "
        "data-busy-message='Uploading reference...' "
        "enctype='multipart/form-data' class='reference-upload'>"
        f"{stage_field}"
        f"{_reference_picker(t('Reference images', lang), 'note', alias_start=alias_start, required=True, lang=lang)}"
        f"<label>{t('Attach to', lang)}<select name='target'>{upload_options}</select></label>"
        f"<label>{t('Label', lang)}<input name='label' placeholder='protagonist face, apartment mood, prop detail'></label>"
        f"<label class='full'>{t('Note', lang)}<textarea name='note' placeholder='Use this face for the protagonist, keep the jacket shape, or match this room layout.'></textarea></label>"
        f"<button type='submit'>{t('Upload reference', lang)}</button>"
        "</form>"
    )

    cards = []
    for record in load_reference_manifest(project):
        rel = record.get("path")
        if not rel or not project.path(*str(rel).split("/")).is_file():
            continue
        status = str(record.get("status") or "resolved")
        alias = record.get("alias") or record.get("original_filename") or rel
        target = (
            f"{record.get('target_type', '')}: {record.get('target_id', '')}"
            if record.get("target_type") != "style" else "style: global"
        )
        inference = record.get("inference") if isinstance(record.get("inference"), dict) else {}
        confidence = inference.get("confidence")
        reason = inference.get("reason")
        retarget_form = (
            f"<form method='post' action='/projects/{_u(project.project_id)}/retarget-reference' "
            "class='reference-retarget' data-busy-message='Attaching reference...'>"
            f"{stage_field}"
            f"<input type='hidden' name='reference_id' value='{_h(record.get('id'))}'>"
            f"<label>{t('Attach to', lang)}<select name='target'>{retarget_options}</select></label>"
            f"<button type='submit' class='secondary'>{t('Attach reference', lang)}</button></form>"
        )
        cards.append(
            "<article class='studio-card reference-card'>"
            f"<img src='{media(str(rel))}' alt='{_h(record.get('label') or rel)}'>"
            f"<header><b>{_h(alias)}</b>"
            f"<span class='badge {_status_class(status)}'>{_h(t(status, lang))}</span></header>"
            f"<p><b>{t('Target', lang)}</b>: {_h(target)}</p>"
            f"{'<p><b>' + t('Inference', lang) + '</b>: ' + _h(confidence) + ' · ' + _h(reason) + '</p>' if confidence is not None else ''}"
            f"{'<p>' + _h(record.get('note')) + '</p>' if record.get('note') else ''}"
            f"<p class='muted'>{_h(rel)}</p>"
            f"{retarget_form}"
            "</article>"
        )
    return (
        f"<section id='studio-references' class='studio-subsection'><h3>{t('References', lang)}</h3>"
        f"{form}"
        f"<div class='studio-cards reference-grid'>{''.join(cards) or '<p class=muted>' + t('No uploaded references yet.', lang) + '</p>'}</div>"
        f"<p class='muted'>{t('Raw manifest', lang)}: {_artifact_link(project, 'references/references.json') if project.path('references', 'references.json').is_file() else 'references/references.json'}</p>"
        "</section>"
    )


def _reference_target_options(
    project: Project,
    *,
    include_auto: bool = False,
    lang: str = "en",
) -> list[tuple[str, str]]:
    options = []
    if include_auto:
        options.append(("auto", t("Auto-detect each image", lang)))
    options.append(("style:global", t("Style / global look", lang)))
    chars = project.path("bible", "characters")
    if chars.is_dir():
        for child in sorted(chars.iterdir()):
            cfile = child / "character.json"
            if not cfile.is_file():
                continue
            data = _read_json(cfile) or {}
            name = data.get("name") or child.name
            options.append((f"character:{name}", f"{t('Character', lang)}: {name}"))
    locs = project.path("bible", "locations")
    if locs.is_dir():
        for child in sorted(locs.iterdir()):
            lfile = child / "location.json"
            if not lfile.is_file():
                continue
            data = _read_json(lfile) or {}
            name = data.get("name") or child.name
            options.append((f"location:{name}", f"{t('Location', lang)}: {name}"))
    return options


def _identity_board_html(data: dict[str, Any], board: dict[str, Any]) -> str:
    """Render the identity board as one clear, de-duplicated block.

    Each canonical look rule is shown once (face / body / hair / wardrobe / palette /
    aliases) with the do / don't lists, so the board reads as a single consistency
    sheet instead of repeating the character description.
    """
    def field_rows(label: str, value: Any) -> list[tuple[str, str]]:
        as_dict = coerce_board_dict(value)
        if as_dict is not None:
            out = []
            for key, sub in as_dict.items():
                text = humanize_board_value(sub)
                if text:
                    out.append((key.replace("_", " ").strip().capitalize(), text))
            return out
        text = humanize_board_value(value)
        return [(label, text)] if text else []

    # Fall back to the canonical character description when no separate identity board
    # exists yet (older projects, or a freshly text-revised character), so the same
    # rich face/body/hair/wardrobe rules still read as point form instead of a raw blob.
    desc = coerce_board_dict(data.get("description")) or {}
    raw_rows = [
        ("Face", board.get("canonical_face") or desc.get("canonical_face")),
        ("Body", board.get("canonical_body") or desc.get("canonical_body")),
        ("Hair", board.get("hair") or desc.get("hair_and_grooming") or desc.get("hair")),
        ("Wardrobe", board.get("wardrobe") or data.get("wardrobe") or desc.get("wardrobe_lock")),
        ("Palette", board.get("palette") or data.get("palette") or desc.get("palette")),
    ]
    aliases = board.get("prompt_aliases") or []
    if isinstance(aliases, list) and aliases:
        raw_rows.append(("Aliases", ", ".join(str(a) for a in aliases)))

    seen: set[str] = set()
    items = []
    for label, value in raw_rows:
        for row_label, text in field_rows(label, value):
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            items.append(f"<div class='id-row'><dt>{_h(row_label)}</dt><dd>{_h(text)}</dd></div>")

    do = "".join(f"<li>{_h(item)}</li>" for item in board.get("do", []) if str(item).strip())
    dont = "".join(f"<li>{_h(item)}</li>" for item in board.get("dont", []) if str(item).strip())
    lists = ""
    if do:
        lists += f"<div class='id-rule do'><h5>Do</h5><ul>{do}</ul></div>"
    if dont:
        lists += f"<div class='id-rule dont'><h5>Don't</h5><ul>{dont}</ul></div>"

    if not items and not lists:
        return "<p class='muted'>No identity board yet.</p>"
    grid = f"<dl class='identity-board'>{''.join(items)}</dl>" if items else ""
    rules = f"<div class='id-rules'>{lists}</div>" if lists else ""
    return f"<div class='identity-board-block'><h4>Identity board</h4>{grid}{rules}</div>"


def _render_adjustment_history(project: Project) -> str:
    history_dir = project.path("adjustments")
    if not history_dir.is_dir():
        return "<div class='history'><h4>Adjustment history</h4><p class='muted'>No LLM revisions yet.</p></div>"
    items = []
    for path in sorted(history_dir.glob("*.json"), reverse=True)[:6]:
        record = _read_json(path) or {}
        state = "applied" if record.get("applied") else "failed"
        items.append(
            f"<li><b>{_h(state)}</b> {_h(record.get('artifact_path'))}<br>"
            f"<span class='muted'>{_h(record.get('instruction'))}</span></li>"
        )
    return f"<div class='history'><h4>Adjustment history</h4><ul>{''.join(items)}</ul></div>"


def render_artifact(
    project: Project, rel_path: str, *, notice: str = "", lang: str = "en"
) -> tuple[str, bytes | None, str]:
    path = safe_project_path(project, rel_path)
    if not path.is_file():
        raise FileNotFoundError(rel_path)
    suffix = path.suffix.lower()
    banner = f"<p class=notice>{_h(notice)}</p>" if notice else ""
    if suffix in MEDIA_EXTS:
        media = f"/projects/{_u(project.project_id)}/media?path={_u(rel_path)}"
        tag = (
            f"<video controls src='{media}'></video>"
            if suffix == ".mp4" else
            f"<audio controls src='{media}'></audio>"
            if suffix in (".wav", ".mp3") else
            f"<img src='{media}' alt='{_h(rel_path)}'>"
        )
        return page(rel_path, f"<main><a href='/projects/{_u(project.project_id)}'>{t('Back', lang)}</a>{banner}{tag}</main>", lang=lang), None, "text/html"

    content = path.read_text() if suffix in TEXT_EXTS else ""
    form = (
        f"<main><a href='/projects/{_u(project.project_id)}'>{t('Back', lang)}</a>"
        f"<h1>{_h(rel_path)}</h1>{banner}"
        f"<form method='post' action='/projects/{_u(project.project_id)}/artifact'>"
        f"<input type='hidden' name='path' value='{_h(rel_path)}'>"
        f"<textarea class='artifact-editor' name='content'>{_h(content)}</textarea>"
        f"<button type='submit'>{t('Save Artifact', lang)}</button></form></main>"
    )
    return page(rel_path, form, lang=lang), None, "text/html"


def run_server(root: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    handler = make_handler(root, jobs=_JOBS)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"Studio Agent web app running at http://{host}:{port}")
    httpd.serve_forever()


def make_handler(root: Path, jobs: JobRunner | None = None):
    jobs = jobs or _JOBS

    class Handler(BaseHTTPRequestHandler):
        def _lang(self):
            from .i18n import lang_from_cookie_header
            return lang_from_cookie_header(self.headers.get("Cookie"))

        def do_GET(self):
            try:
                self._handle_get()
            except Exception as exc:  # local tool: show useful errors in the browser
                self._send(500, page("Error", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))

        def do_POST(self):
            self._request_writer_project_id = None
            try:
                self._handle_post()
            except Exception as exc:
                self._send(500, page("Error", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))
            finally:
                project_id = self._request_writer_project_id
                if project_id:
                    jobs._release_project_writer(project_id)
                    self._request_writer_project_id = None

        def _handle_get(self):
            url = urllib.parse.urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            qs = urllib.parse.parse_qs(url.query)
            if not parts:
                self._send(200, render_index(
                    root,
                    jobs=jobs,
                    notice=qs.get("notice", [""])[0],
                    lang=self._lang(),
                ))
                return
            if parts == ["jobs.json"]:
                self._send_json(200, {"jobs": jobs.snapshot(root), "html": _render_job_rows(jobs.snapshot(root))})
                return
            if len(parts) >= 2 and parts[0] == "projects":
                project = Project.load(root / parts[1])
                if len(parts) == 2:
                    self._send(200, render_project(project, stage=qs.get("stage", [None])[0], lang=self._lang()))
                    return
                if parts[2] == "state.json":
                    self._send_json(200, project_state(project))
                    return
                rel = qs.get("path", [""])[0]
                if parts[2] == "artifact":
                    text, data, ctype = render_artifact(
                        project, rel, notice=qs.get("notice", [""])[0], lang=self._lang()
                    )
                    self._send(200, text, ctype)
                    return
                if parts[2] == "media":
                    self._send_file(safe_project_path(project, rel))
                    return
            self._send(404, page("Not found", "<main>Not found</main>", lang=self._lang()))

        def _handle_post(self):
            url = urllib.parse.urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            fields = self._fields()
            if parts == ["runs"]:
                notice = self._start_run(fields)
                self._send(200, render_index(root, jobs=jobs, notice=notice, lang=self._lang()))
                return
            if len(parts) < 3 or parts[0] != "projects":
                self._redirect("/")
                return

            action = parts[2]
            if action == "delete":
                project_id = urllib.parse.unquote(parts[1])
                if jobs.has_active_project_job(project_id):
                    notice = (
                        f"Project {project_id} is currently generating; "
                        "wait for the job to finish before deleting it."
                    )
                    self._redirect(f"/?notice={_u(notice)}")
                    return
                try:
                    with jobs.project_writer(project_id):
                        delete_project(root, project_id)
                        notice = f"Deleted project: {project_id}"
                except ProjectBusyError as exc:
                    self._send(409, page("Project busy", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))
                    return
                except (FileNotFoundError, ValueError):
                    notice = f"Project could not be deleted: {project_id}"
                self._redirect(f"/?notice={_u(notice)}")
                return

            project_id = urllib.parse.unquote(parts[1])
            if action in ("resume", "resume-auto"):
                try:
                    jobs.start_resume(
                        root,
                        project_id=project_id,
                        auto=(action == "resume-auto"),
                    )
                except ProjectBusyError as exc:
                    self._send(409, page("Project busy", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))
                    return
                self._redirect(f"/projects/{_u(project_id)}")
                return

            try:
                jobs._acquire_project_writer(project_id)
            except ProjectBusyError as exc:
                self._send(409, page("Project busy", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))
                return
            self._request_writer_project_id = project_id
            project = Project.load(root / parts[1])
            notice = ""
            if action == "answer-decision":
                stage = fields.get("stage", [""])[0].strip()
                choice = fields.get("choice", [""])[0].strip()
                resolved = resolve_decision(project, stage=stage, choice=choice)
                self._dispatch_run(
                    project.project_id,
                    f"Decision saved: {resolved['resolution']['value']}. Regenerating.",
                )
                return
            elif action == "refresh-knowledge":
                purpose = fields.get("purpose", [""])[0].strip()
                target = fields.get("target", [""])[0].strip()
                try:
                    refreshed = refresh_knowledge_target(
                        project,
                        purpose=purpose,
                        target=target,
                    )
                    self._dispatch_run(
                        project.project_id,
                        f"Refreshed {purpose} knowledge for {target}; archived "
                        f"{refreshed.archived} dependent artifact(s). Regenerating.",
                    )
                    return
                except Exception as exc:
                    notice = f"Refresh failed for {purpose}/{target}: {exc}"
            elif action == "prompt-chat":
                gate = fields.get("gate", [""])[0].strip()
                expected = _active_prompt_kind(project)
                message = fields.get("message", [""])[0].strip()
                shot_id = fields.get("shot_id", [""])[0].strip()
                apply_to_all = "apply_to_all" in fields
                if gate != expected:
                    notice = f"Prompt gate mismatch: expected {expected or 'none'}, got {gate or 'none'}."
                elif not message:
                    notice = "Tell the project-aware LLM what to change."
                else:
                    try:
                        result = revise_prompt_turn(
                            project,
                            gate=gate,
                            shot_id=shot_id,
                            message=message,
                            apply_to_all=apply_to_all,
                            providers=_revision_providers(project),
                        )
                        notice = f"{message} — {result.assistant_message} {result.disclosure}"
                    except Exception as exc:
                        notice = f"Prompt revision failed: {exc}"
            elif action == "confirm-prompts":
                kind = fields.get("kind", [""])[0].strip()
                expected = _active_prompt_kind(project)
                if kind != expected:
                    notice = f"Prompt gate mismatch: expected {expected or 'none'}, got {kind or 'none'}."
                else:
                    try:
                        from . import cli
                        previous_stage = project.current_stage
                        next_stage = cli._machine().approve(project)
                        if next_stage is None:
                            notice = "Prompts confirmed. Project done."
                        elif next_stage == previous_stage:
                            notice = "The prompt stage must finish before confirmation."
                        else:
                            self._dispatch_run(
                                project.project_id,
                                f"Confirmed all {kind} prompts. Running {next_stage}.",
                            )
                            return
                    except Exception as exc:
                        notice = f"Prompt confirmation failed: {exc}"
            elif action == "approve":
                from . import cli
                next_stage = cli._machine().approve(project)
                if next_stage is None:
                    self._redirect(f"/projects/{_u(project.project_id)}?notice={_u('Approved. Project done.')}")
                    return
                if next_stage == "bible":
                    # Pause before the bible (do not auto-run): the characters and
                    # locations are now named, so the user may want to upload a reference
                    # per character/location before the sheets generate. They press Resume
                    # (继续) when ready.
                    self._redirect(
                        f"/projects/{_u(project.project_id)}?notice="
                        + _u("Approved. Upload any character/location references, then press Resume to build the bible.")
                    )
                    return
                self._dispatch_run(project.project_id, f"Approved. Running {next_stage}.")
                return
            elif action == "regenerate":
                from . import cli
                shot_id = fields.get("shot", [""])[0].strip()
                shot = cli._find_shot(project, shot_id)
                if shot is None:
                    notice = f"Shot not found: {shot_id}"
                else:
                    block = maybe_apply_video_model(project, cli.load_config(), fields)
                    if block:
                        notice = block
                    else:
                        try:
                            regenerate_shot_video(project, shot_id)
                            self._dispatch_run(
                                project.project_id,
                                f"Generating a replacement candidate for {shot_id}; "
                                "the accepted clip stays live until it succeeds.",
                            )
                            return
                        except Exception as exc:
                            notice = f"Regeneration setup failed for {shot_id}: {exc}"
            elif action == "artifact":
                rel = fields.get("path", [""])[0]
                content = fields.get("content", [""])[0]
                path = safe_project_path(project, rel)
                if path.suffix.lower() not in TEXT_EXTS:
                    raise ValueError("only text artifacts can be edited in the web app")
                path.write_text(content)
                # A hand-edit replaces the artifact, so its downstream derivatives are now
                # stale: archive them and flip the dependent stages to pending (same cascade
                # the LLM "Revise" path uses) so the user can re-run forward from here.
                invalidated = mark_artifact_revised(project, rel)
                pending = ", ".join(invalidated) or "none"
                self._redirect(
                    f"/projects/{_u(project.project_id)}/artifact?path={_u(rel)}"
                    f"&notice={_u(f'Saved {rel}. Downstream pending: {pending}.')}"
                )
                return
            elif action == "adjust-artifact":
                rel = fields.get("path", [""])[0].strip()
                instruction = fields.get("instruction", [""])[0].strip()
                if not instruction:
                    notice = "Tell the LLM what to change before pressing Revise."
                elif not rel:
                    notice = "No editable artifact is selected for this stage."
                else:
                    try:
                        result = apply_artifact_revision(
                            project,
                            rel,
                            instruction,
                            _revision_providers(project),
                        )
                        invalidated = ", ".join(result.invalidated) or "none"
                        notice = f"Revised {rel}. Downstream pending: {invalidated}."
                    except Exception as exc:
                        notice = f"Revision failed for {rel}: {exc}"
            elif action == "style-feedback":
                feedback = fields.get("feedback", [""])[0].strip()
                if not feedback:
                    notice = "Add feedback before revising the style."
                else:
                    style_input = dict(project.model_config.get("style_input") or {})
                    prior = str(style_input.get("feedback") or "").strip()
                    style_input["feedback"] = f"{prior}\n{feedback}".strip() if prior else feedback
                    project.model_config["style_input"] = style_input
                    archive_paths(project, ["bible/style_sample.png"], reason="style-feedback")
                    project.set_stage_status("style", "pending")
                    project.current_stage = "style"
                    project.save()
                    self._dispatch_run(project.project_id, "Refining style; regenerating sample.")
                    return
            elif action == "keyframe-feedback":
                shot_id = fields.get("shot_id", [""])[0].strip()
                feedback = fields.get("feedback", [""])[0].strip()
                if not feedback:
                    notice = "Add feedback before revising the keyframe prompt."
                else:
                    try:
                        result = revise_keyframe_from_feedback(
                            project,
                            shot_id,
                            feedback,
                            _revision_providers(project),
                        )
                        self._dispatch_run(
                            project.project_id,
                            f"Revised keyframe prompt for {shot_id}; removed "
                            f"{result.removed} artifact(s). Regenerating.",
                        )
                        return
                    except Exception as exc:
                        notice = f"Keyframe feedback failed for {shot_id}: {exc}"
            elif action == "video-feedback":
                shot_id = fields.get("shot_id", [""])[0].strip()
                feedback = fields.get("feedback", [""])[0].strip()
                if not feedback:
                    notice = "Add feedback before revising the video prompt."
                else:
                    try:
                        result = revise_video_from_feedback(
                            project,
                            shot_id,
                            feedback,
                            _revision_providers(project),
                        )
                        # Revise only — the paid clip is re-rendered only when the user
                        # clicks "Regenerate this shot" (invariant #7). No run dispatched.
                        notice = (
                            f"Revised the motion prompt for {shot_id}. "
                            "Review it, then click Regenerate this shot to re-render."
                            if result.applied
                            else f"Could not revise the video prompt for {shot_id}: {result.error}"
                        )
                    except Exception as exc:
                        notice = f"Video feedback failed for {shot_id}: {exc}"
            elif action == "delete-clip":
                shot_id = fields.get("shot_id", [""])[0].strip()
                try:
                    result = delete_clip_shot(project, shot_id)
                    notice = (
                        f"Deleted {shot_id}; archived {result.removed} artifact(s). "
                        "Review the updated shot plan before continuing."
                    )
                except Exception as exc:
                    notice = f"Delete clip failed for {shot_id}: {exc}"
            elif action == "add-shot":
                after_shot_id = fields.get("after_shot_id", [""])[0].strip()
                description = fields.get("description", [""])[0].strip()
                try:
                    result = add_shot(
                        project,
                        description=description,
                        after_shot_id=after_shot_id,
                        providers=_revision_providers(project),
                    )
                    self._dispatch_run(
                        project.project_id,
                        f"Added {result.shot_id}; drafting its prompt, then pausing at "
                        f"the {result.planning_stage} gate. Re-confirm the prompt batch "
                        "before the paid stages — untouched shots stay approved.",
                    )
                    return
                except Exception as exc:
                    notice = f"Add shot failed: {exc}"
            elif action == "set-clip-duration":
                shot_id = fields.get("shot_id", [""])[0].strip()
                try:
                    raw_seconds = fields.get("seconds", [""])[0].strip()
                    result = set_clip_duration(project, shot_id, raw_seconds or None)
                    label = f"{raw_seconds}s" if raw_seconds else "the video model's default length"
                    notice = (
                        f"Updated {shot_id} to {label}; archived "
                        f"{result.removed} dependent artifact(s)."
                    )
                except Exception as exc:
                    notice = f"Duration update failed for {shot_id}: {exc}"
            elif action == "regenerate-asset":
                kind = fields.get("kind", [""])[0].strip()
                rel = fields.get("rel_path", [""])[0].strip()
                shot_id = fields.get("shot_id", [""])[0].strip()
                try:
                    if kind == "bible_asset":
                        regen = regenerate_bible_asset(project, rel)
                        downstream = [s for s in regen.invalidated if s != "bible"]
                        self._dispatch_run(
                            project.project_id,
                            f"Regenerated {rel}. Review the updated bible, then approve. "
                            f"Downstream pending: {', '.join(downstream) or 'none'}.",
                        )
                        return
                    elif kind == "keyframe":
                        regen = regenerate_keyframe(project, shot_id)
                        self._dispatch_run(
                            project.project_id,
                            f"Generating a replacement keyframe candidate for {shot_id}; "
                            "the accepted image stays live until it succeeds.",
                        )
                        return
                    elif kind == "shot_video":
                        regen = regenerate_shot_video(project, shot_id)
                        self._dispatch_run(
                            project.project_id,
                            f"Generating a replacement video candidate for {shot_id}; "
                            "the accepted clip stays live until it succeeds.",
                        )
                        return
                    else:
                        notice = f"Unknown regeneration action: {kind}"
                except Exception as exc:
                    notice = f"Regeneration setup failed: {exc}"
            elif action == "regenerate-bible-text":
                kind = fields.get("kind", [""])[0].strip()
                slug = fields.get("slug", [""])[0].strip()
                instruction = fields.get("instruction", [""])[0].strip()
                if not instruction:
                    notice = "Add a comment before regenerating the text."
                else:
                    try:
                        result = regenerate_bible_text(
                            project, kind, slug, instruction, _revision_providers(project)
                        )
                        if result.needs_resume:
                            self._dispatch_run(
                                project.project_id,
                                f"Revised {result.rel} text. Review it, then approve to regenerate the image.",
                            )
                            return
                        notice = f"Revised {result.rel} text. Review it, then approve to regenerate the image."
                    except Exception as exc:
                        notice = f"Text regeneration failed for {kind}/{slug}: {exc}"
            elif action == "upload-reference":
                target = fields.get("target", ["auto"])[0]
                if target == "auto":
                    target_type, target_id = "", ""
                elif ":" in target:
                    target_type, target_id = target.split(":", 1)
                else:
                    target_type, target_id = target, ""
                uploads = _file_list(getattr(self, "_files", {}), "reference_images")
                if not uploads:
                    raise ValueError("choose one or more images to upload")
                providers = _revision_providers(project)
                records = save_reference_intake_uploads(
                    project,
                    uploads,
                    analyzer=providers.reference_analyzer,
                    target_type=target_type,
                    target_id=target_id,
                    label=fields.get("label", [""])[0],
                    note=fields.get("note", [""])[0],
                )
                resolved_targets = {
                    (str(record.get("target_type") or ""), str(record.get("target_id") or ""))
                    for record in records
                    if record.get("status") == "resolved" and record.get("target_type")
                }
                target_type, target_id = (
                    next(iter(resolved_targets)) if len(resolved_targets) == 1 else ("", "")
                )
                invalidated = mark_reference_upload_revised(
                    project,
                    target_type=target_type,
                    target_id=target_id,
                )
                pending = ", ".join(invalidated) or "none"
                aliases = [str(record.get("alias") or record.get("original_filename")) for record in records]
                alias_range = aliases[0] if len(aliases) == 1 else f"{aliases[0]} to {aliases[-1]}"
                notice = (
                    f"Uploaded {len(records)} reference{'s' if len(records) != 1 else ''} "
                    f"({alias_range}). Downstream pending: {pending}."
                )
            elif action == "retarget-reference":
                target = fields.get("target", ["style:global"])[0]
                if ":" in target:
                    target_type, target_id = target.split(":", 1)
                else:
                    target_type, target_id = target, ""
                reference_id = fields.get("reference_id", [""])[0].strip()
                record = retarget_reference(project, reference_id, target_type, target_id)
                invalidated = mark_reference_upload_revised(
                    project,
                    target_type=target_type,
                    target_id=target_id,
                )
                pending = ", ".join(invalidated) or "none"
                notice = f"Attached reference {record['path']}. Downstream pending: {pending}."
            stage_sel = fields.get("stage", [""])[0].strip() or None
            self._send(200, render_project(
                Project.load(project.dir), notice=notice, stage=stage_sel, lang=self._lang()
            ))

        def _start_run(self, fields):
            profile_name = CUSTOM_MODEL_PROFILE
            model_parts = {
                kind: fields.get(f"{kind}_option", ["fake"])[0]
                for kind in MODEL_OPTION_KINDS
            }
            idea = fields.get("idea", [""])[0].strip()
            if not idea:
                return "Add an idea before starting a run."
            run_mode = fields.get("run_mode", [None])[0]
            auto = (run_mode == "auto") if run_mode is not None else ("auto" in fields)
            creative_inputs = {
                key: value
                for key in (
                    "tone",
                    "visual_world",
                    "camera_language",
                    "light_texture",
                    "character_treatment",
                )
                if (value := fields.get(key, [""])[0].strip())
            }
            hard_avoidances = [
                line.strip()
                for line in fields.get("hard_avoidances", [""])[0].splitlines()
                if line.strip()
            ]
            if hard_avoidances:
                creative_inputs["hard_avoidances"] = hard_avoidances
            music_enabled = "music" in fields
            music_mood = fields.get("music_mood", [""])[0].strip()
            job = jobs.start_run(
                root,
                idea=idea,
                profile_name=profile_name,
                style_name=fields.get("style", [""])[0] or None,
                style_description=fields.get("style_description", [""])[0].strip(),
                style_image=next(iter(_file_list(getattr(self, "_files", {}), "style_image")), None),
                format_name=fields.get("format", [""])[0] or None,
                language=self._lang(),
                auto=auto,
                length=fields.get("length", ["auto"])[0] or "auto",
                clip_plan=fields.get("clip_plan", ["default"])[0] or "default",
                clips=fields.get("clips", [""])[0].strip(),
                clip_seconds=fields.get("clip_seconds", [""])[0].strip(),
                model_parts=model_parts,
                reference_uploads=_file_list(getattr(self, "_files", {}), "reference_images"),
                reference_note=fields.get("reference_note", [""])[0],
                creative_inputs=creative_inputs,
                music_enabled=music_enabled,
                music_mood=music_mood,
            )
            mode = "auto approve" if auto else "review each stage"
            return f"Started {profile_name} ({mode}): job {job.id}"

        def _fields(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            self._files = {}
            if content_type.lower().startswith("multipart/form-data"):
                fields, files = _parse_multipart_form(raw, content_type)
                self._files = files
                return fields
            return urllib.parse.parse_qs(raw.decode("utf-8"))

        def _send_file(self, path: Path):
            if not path.is_file():
                self._send(404, "not found", "text/plain")
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            stat = path.stat()
            # Regenerated assets (bible reference sheets, keyframes, clips) reuse a
            # stable media URL, so without revalidation the browser keeps showing the
            # old image. Force a conditional revalidation keyed on the file's mtime/size
            # so a fresh generation is fetched while unchanged files still 304 cheaply.
            etag = f'"{int(stat.st_mtime_ns)}-{stat.st_size}"'
            last_modified = self.date_time_string(int(stat.st_mtime))
            if self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.end_headers()
            self.wfile.write(path.read_bytes())

        def _send(self, status: int, text: str, ctype: str = "text/html"):
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: int, data: dict):
            text = json.dumps(data)
            self._send(status, text, "application/json")

        def _dispatch_run(self, project_id: str, notice: str) -> None:
            # Release the writer lock this POST holds so the background job can acquire it,
            # then run the next stage in the background (auto=False pauses at its gate) and
            # redirect immediately — the page polls /jobs.json for progress.
            if self._request_writer_project_id:
                jobs._release_project_writer(self._request_writer_project_id)
                self._request_writer_project_id = None
            try:
                jobs.start_resume(root, project_id=project_id, auto=False)
            except ProjectBusyError as exc:
                self._send(409, page("Project busy", f"<main><pre>{_h(str(exc))}</pre></main>", lang=self._lang()))
                return
            self._redirect(f"/projects/{_u(project_id)}?notice={_u(notice)}")

        def _redirect(self, target: str):
            self.send_response(303)
            self.send_header("Location", target)
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002 - inherited API
            return

    return Handler


def _parse_multipart_form(
    raw: bytes,
    content_type: str,
) -> tuple[dict[str, list[str]], dict[str, UploadedFormFile | list[UploadedFormFile]]]:
    boundary = _content_type_param(content_type, "boundary")
    if not boundary:
        raise ValueError("multipart form is missing a boundary")
    marker = b"--" + boundary.encode("utf-8")
    fields: dict[str, list[str]] = {}
    files: dict[str, UploadedFormFile | list[UploadedFormFile]] = {}
    for part in raw.split(marker):
        if not part:
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part in (b"--", b"--\r\n"):
            continue
        if part.endswith(b"--\r\n"):
            part = part[:-4]
        elif part.endswith(b"--"):
            part = part[:-2]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if b"\r\n\r\n" not in part:
            continue
        header_blob, body = part.split(b"\r\n\r\n", 1)
        headers = _part_headers(header_blob)
        disposition = headers.get("content-disposition", "")
        _, params = _parse_header_params(disposition)
        name = params.get("name", "")
        if not name:
            continue
        filename = params.get("filename")
        if filename is not None:
            upload = UploadedFormFile(
                filename=filename,
                content_type=headers.get("content-type", "application/octet-stream"),
                data=body,
            )
            existing = files.get(name)
            if existing is None:
                files[name] = upload
            elif isinstance(existing, list):
                existing.append(upload)
            else:
                files[name] = [existing, upload]
        else:
            fields.setdefault(name, []).append(body.decode("utf-8", errors="replace"))
    return fields, files


def _file_list(
    files: dict[str, UploadedFormFile | list[UploadedFormFile]],
    name: str,
) -> list[UploadedFormFile]:
    value = files.get(name)
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [item for item in items if item.filename and item.data]


def _part_headers(blob: bytes) -> dict[str, str]:
    headers = {}
    for raw_line in blob.decode("utf-8", errors="replace").split("\r\n"):
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def _content_type_param(header: str, key: str) -> str:
    _main, params = _parse_header_params(header)
    return params.get(key, "")


def _parse_header_params(header: str) -> tuple[str, dict[str, str]]:
    pieces = [part.strip() for part in str(header or "").split(";")]
    main = pieces[0].lower() if pieces else ""
    params = {}
    for piece in pieces[1:]:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        params[key.strip().lower()] = value
    return main, params


def _client_i18n_script(lang: str) -> str:
    """Hand the browser the active language plus a translation map generated
    from the one Python catalog. The client uses it only for strings it injects
    at runtime (busy overlays, the refresh notice, the delete-confirm fallback);
    the static DOM is already server-translated. ``en`` ships an empty map."""
    mapping = dict(UI_STRINGS) if lang == "zh" else {}
    payload = json.dumps({"lang": lang, "strings": mapping}, ensure_ascii=False)
    return f"<script>window.STUDIO_I18N={payload};</script>"


def page(title: str, body: str, lang: str = "en") -> str:
    return (
        f"<!doctype html><html lang='{_h(lang)}'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_h(t(title, lang))}</title><style>{CSS}</style></head><body>{body}"
        "<div class='lightbox' data-lightbox hidden aria-hidden='true'>"
        "<button type='button' class='lightbox-close' data-lightbox-close aria-label='Close'>&times;</button>"
        "<img alt='' data-lightbox-img></div>"
        f"{_client_i18n_script(lang)}"
        f"<script>{BUSY_JS}</script></body></html>"
    )


_DONE_STATES = {"approved", "done", "complete", "completed", "passed"}


def _status_class(status: Any) -> str:
    """Map a project/job status to a badge CSS class."""
    return {
        "done": "done", "complete": "done", "completed": "done",
        "in_progress": "running", "running": "running",
        "paused": "paused", "queued": "queued",
        "failed": "failed", "error": "failed",
    }.get(str(status or "").lower(), "none")


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _project_version(project: Project) -> str:
    if not project.json_path.is_file():
        return "0"
    return str(project.json_path.stat().st_mtime_ns)


def _h(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _u(value: str) -> str:
    return urllib.parse.quote(value, safe="")


BUSY_JS = """
function studioT(text) {
  var trimmed = String(text || '').trim();
  if (!trimmed) return trimmed;
  var map = (window.STUDIO_I18N && window.STUDIO_I18N.strings) || {};
  return map[trimmed] || trimmed;
}

var STUDIO_MODEL_PREFERENCES_KEY = 'studioModelPreferences:v1';

function studioModelPreferenceSelects() {
  return document.querySelectorAll('select[data-model-preference]');
}

function studioRestoreModelPreferences() {
  try {
    var raw = localStorage.getItem(STUDIO_MODEL_PREFERENCES_KEY);
    if (!raw) return;
    var saved = JSON.parse(raw);
    if (!saved || saved.version !== 1 || !saved.models || typeof saved.models !== 'object') return;
    studioModelPreferenceSelects().forEach(function (select) {
      var kind = select.getAttribute('data-model-preference');
      var value = saved.models[kind];
      if (typeof value !== 'string') return;
      var exists = Array.prototype.some.call(select.options, function (option) {
        return option.value === value;
      });
      if (exists) select.value = value;
    });
  } catch (e) {}
}

function studioSaveModelPreferences() {
  try {
    var models = {};
    studioModelPreferenceSelects().forEach(function (select) {
      models[select.getAttribute('data-model-preference')] = select.value;
    });
    localStorage.setItem(STUDIO_MODEL_PREFERENCES_KEY, JSON.stringify({
      version: 1,
      models: models
    }));
  } catch (e) {}
}

var STUDIO_SCROLL_KEY = 'studioScrollRestore:v1';
var STUDIO_SCROLL_MAX_AGE_MS = 6 * 60 * 60 * 1000;

function studioScrollAnchor() {
  if (!document.elementFromPoint || window.innerWidth < 1 || window.innerHeight < 1) return null;
  var x = Math.max(0, Math.min(window.innerWidth - 1, window.innerWidth / 2));
  var y = Math.max(0, Math.min(window.innerHeight - 1, window.innerHeight / 2));
  var element = document.elementFromPoint(x, y);
  return element && element.closest ? element.closest('[id]') : null;
}

function studioRememberScroll() {
  try {
    var anchor = studioScrollAnchor();
    var snapshot = {
      path: window.location.pathname,
      x: Number(window.scrollX || 0),
      y: Number(window.scrollY || 0),
      anchorId: anchor ? anchor.id : '',
      anchorOffset: anchor ? anchor.getBoundingClientRect().top : null,
      savedAt: Date.now()
    };
    sessionStorage.setItem(STUDIO_SCROLL_KEY, JSON.stringify(snapshot));
  } catch (e) {}
}

function studioRestoreScroll() {
  var snapshot;
  try {
    var raw = sessionStorage.getItem(STUDIO_SCROLL_KEY);
    if (!raw) return;
    sessionStorage.removeItem(STUDIO_SCROLL_KEY);
    snapshot = JSON.parse(raw);
  } catch (e) {
    return;
  }
  if (!snapshot || snapshot.path !== window.location.pathname) return;
  if (Date.now() - Number(snapshot.savedAt || 0) > STUDIO_SCROLL_MAX_AGE_MS) return;

  function restore() {
    var x = Number(snapshot.x || 0);
    var y = Number(snapshot.y || 0);
    if (snapshot.anchorId && snapshot.anchorOffset !== null) {
      var anchor = document.getElementById(snapshot.anchorId);
      if (anchor) {
        y = Number(window.scrollY || 0)
          + anchor.getBoundingClientRect().top
          - Number(snapshot.anchorOffset || 0);
      }
    }
    window.scrollTo(x, Math.max(0, y));
  }

  requestAnimationFrame(function () { requestAnimationFrame(restore); });
  window.addEventListener('load', restore, { once: true });
}

function studioInitProjectDeleteDialog() {
  var dialog = document.querySelector('[data-delete-dialog]');
  if (!dialog) return;
  var form = dialog.querySelector('[data-delete-form]');
  var projectName = dialog.querySelector('[data-delete-project-name]');
  var cancel = dialog.querySelector('[data-delete-cancel]');
  if (!form || !projectName || !cancel) return;

  document.querySelectorAll('[data-delete-project]').forEach(function (button) {
    button.addEventListener('click', function () {
      var projectId = button.getAttribute('data-project-id') || '';
      form.action = '/projects/' + encodeURIComponent(projectId) + '/delete';
      projectName.textContent = projectId;
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
        cancel.focus();
        return;
      }
      var question = studioT('Permanently delete project?');
      if (window.confirm(question + '\\n\\n' + projectId)) form.requestSubmit();
    });
  });

  cancel.addEventListener('click', function () { dialog.close(); });
  dialog.addEventListener('click', function (event) {
    if (event.target !== dialog) return;
    var rect = dialog.getBoundingClientRect();
    var inside = event.clientX >= rect.left && event.clientX <= rect.right
      && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) dialog.close();
  });
}

function studioInitBibleDetailDialogs() {
  document.querySelectorAll('[data-bible-detail-open]').forEach(function (button) {
    button.addEventListener('click', function () {
      var id = button.getAttribute('data-bible-detail-open');
      var dialog = id ? document.getElementById(id) : null;
      if (!dialog) return;
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    });
  });
  document.querySelectorAll('.bible-detail-dialog').forEach(function (dialog) {
    dialog.addEventListener('click', function (event) {
      if (event.target !== dialog) return;
      var rect = dialog.getBoundingClientRect();
      var inside = event.clientX >= rect.left && event.clientX <= rect.right
        && event.clientY >= rect.top && event.clientY <= rect.bottom;
      if (!inside) dialog.close();
    });
  });
}

document.addEventListener('click', function (event) {
  var button = event.target.closest && event.target.closest('[data-lang-toggle] button');
  if (!button) return;
  var lang = button.getAttribute('data-lang') === 'zh' ? 'zh' : 'en';
  try { localStorage.setItem('studioLang', lang); } catch (e) {}
  document.cookie = 'studioLang=' + lang + ';path=/;max-age=31536000;samesite=lax';
  window.location.reload();
});

document.addEventListener('change', function (event) {
  var target = event.target;
  if (!target || !target.matches || !target.matches('select[data-model-preference]')) return;
  studioSaveModelPreferences();
});

document.addEventListener('input', function (event) {
  var form = event.target.closest && event.target.closest('form');
  if (form) form.dataset.dirty = 'true';
});

function studioInsertReferenceAlias(textarea, alias) {
  if (!textarea) return;
  var value = textarea.value || '';
  var start = typeof textarea.selectionStart === 'number' ? textarea.selectionStart : value.length;
  var end = typeof textarea.selectionEnd === 'number' ? textarea.selectionEnd : start;
  var leftSpace = start > 0 && !/\\s/.test(value.charAt(start - 1)) ? ' ' : '';
  var rightSpace = end < value.length && !/\\s/.test(value.charAt(end)) ? ' ' : '';
  var insertion = leftSpace + alias + rightSpace;
  if (typeof textarea.setRangeText === 'function') {
    textarea.setRangeText(insertion, start, end, 'end');
  } else {
    textarea.value = value.slice(0, start) + insertion + value.slice(end);
  }
  textarea.focus();
  textarea.dispatchEvent(new Event('input', { bubbles: true }));
}

function studioInitReferencePickers() {
  document.querySelectorAll('[data-reference-picker]').forEach(function (picker) {
    if (picker.dataset.referencePickerReady === 'true') return;
    picker.dataset.referencePickerReady = 'true';
    var input = picker.querySelector('[data-reference-files]');
    var preview = picker.querySelector('[data-reference-preview]');
    var form = picker.closest('form');
    var note = form && form.elements.namedItem(picker.dataset.referenceNote || '');
    var aliasStart = parseInt(picker.dataset.aliasStart || '1', 10) || 1;
    var selectedFiles = [];
    var objectUrls = [];
    if (!input || !preview) return;

    function releaseObjectUrls() {
      objectUrls.forEach(function (url) { URL.revokeObjectURL(url); });
      objectUrls = [];
    }

    function syncFileInput() {
      var transfer = new DataTransfer();
      selectedFiles.forEach(function (file) { transfer.items.add(file); });
      input.files = transfer.files;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function renderPreviews() {
      releaseObjectUrls();
      preview.replaceChildren();
      preview.hidden = selectedFiles.length === 0;
      selectedFiles.forEach(function (file, index) {
        var alias = '@image' + (aliasStart + index);
        var card = document.createElement('article');
        card.className = 'reference-preview-card';

        var image = document.createElement('img');
        var objectUrl = URL.createObjectURL(file);
        objectUrls.push(objectUrl);
        image.src = objectUrl;
        image.alt = file.name;
        card.appendChild(image);

        var remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'reference-remove';
        remove.textContent = '×';
        remove.setAttribute('aria-label', 'Remove ' + file.name);
        remove.title = 'Remove ' + file.name;
        remove.addEventListener('click', function () {
          selectedFiles.splice(index, 1);
          syncFileInput();
          renderPreviews();
        });
        card.appendChild(remove);

        var meta = document.createElement('div');
        meta.className = 'reference-preview-meta';
        var aliasButton = document.createElement('button');
        aliasButton.type = 'button';
        aliasButton.className = 'reference-alias';
        aliasButton.textContent = alias;
        aliasButton.title = 'Insert ' + alias + ' into note';
        aliasButton.addEventListener('click', function () {
          studioInsertReferenceAlias(note, alias);
        });
        var filename = document.createElement('span');
        filename.className = 'reference-filename';
        filename.textContent = file.name;
        filename.title = file.name;
        meta.appendChild(aliasButton);
        meta.appendChild(filename);
        card.appendChild(meta);
        preview.appendChild(card);
      });
    }

    input.addEventListener('change', function () {
      // Accumulate across picks so the user can build up MANY references (one at a
      // time or in batches), not just the most recent selection. Dedupe identical
      // files, then push the combined list back onto the input so all of them submit.
      var keys = selectedFiles.map(function (f) { return f.name + ':' + f.size + ':' + f.lastModified; });
      Array.from(input.files || []).forEach(function (file) {
        var key = file.name + ':' + file.size + ':' + file.lastModified;
        if (keys.indexOf(key) === -1) { selectedFiles.push(file); keys.push(key); }
      });
      syncFileInput();
      renderPreviews();
    });
    window.addEventListener('beforeunload', releaseObjectUrls);
  });
}

function studioShowRefreshNotice() {
  var notice = document.querySelector('.refresh-notice');
  if (!notice) {
    notice = document.createElement('div');
    notice.className = 'refresh-notice';
    notice.innerHTML = '<span></span><button type="button"></button>';
    document.body.appendChild(notice);
    notice.querySelector('button').addEventListener('click', function () {
      studioRememberScroll();
      window.location.reload();
    });
  }
  notice.querySelector('span').textContent = studioT('New results are ready.');
  notice.querySelector('button').textContent = studioT('Refresh now');
}

function studioHasUnsavedInput() {
  var active = document.activeElement;
  if (active && /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName || '')) return true;
  return !!document.querySelector('form[data-dirty="true"]');
}

function studioStartProjectPolling() {
  var stateUrl = window.STUDIO_PROJECT_STATE_URL;
  var currentVersion = window.STUDIO_PROJECT_VERSION;
  if (!stateUrl || !currentVersion) return;
  var polling = false;
  async function pollProjectState() {
    if (polling || document.body.classList.contains('is-busy')) return;
    polling = true;
    try {
      var response = await fetch(stateUrl + '?t=' + Date.now(), { cache: 'no-store' });
      if (!response.ok) return;
      var state = await response.json();
      if (!state.version || state.version === currentVersion) return;
      if (studioHasUnsavedInput()) {
        studioShowRefreshNotice();
        return;
      }
      document.body.classList.add('is-refreshing');
      studioRememberScroll();
      window.location.reload();
    } catch (e) {
      return;
    } finally {
      polling = false;
    }
  }
  window.setInterval(pollProjectState, 2500);
}

document.addEventListener('submit', studioRememberScroll, true);

document.addEventListener('submit', function (event) {
  var form = event.target;
  if (!form || !form.matches || !form.matches('form[data-busy-message]')) return;
  if (form.dataset.submitting === 'true') {
    event.preventDefault();
    return;
  }
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }
  event.preventDefault();
  var message = form.getAttribute('data-busy-message') || 'Working...';
  form.dataset.submitting = 'true';
  document.body.classList.add('is-busy');
  var button = event.submitter || form.querySelector('button[type="submit"], button:not([type])');
  if (button && button.name) {
    var hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = button.name;
    hidden.value = button.value;
    form.appendChild(hidden);
  }
  var displayMessage = studioT(message);
  if (button) {
    button.dataset.originalText = button.textContent;
    button.textContent = displayMessage;
    button.setAttribute('aria-busy', 'true');
    button.disabled = true;
  }
  var note = form.querySelector('.busy-note');
  if (!note) {
    note = document.createElement('p');
    note.className = 'busy-note';
    form.appendChild(note);
  }
  note.textContent = displayMessage;
  requestAnimationFrame(function () {
    window.setTimeout(function () {
      HTMLFormElement.prototype.submit.call(form);
    }, 0);
  });
});
function studioInitLightbox() {
  var box = document.querySelector('[data-lightbox]');
  if (!box) return;
  var img = box.querySelector('[data-lightbox-img]');
  function closeBox() {
    box.hidden = true;
    box.setAttribute('aria-hidden', 'true');
    img.removeAttribute('src');
    document.body.classList.remove('lightbox-open');
  }
  document.addEventListener('click', function (event) {
    var target = event.target;
    if (!target) return;
    // A click on the overlay backdrop, the open image, or the close button dismisses it.
    if (target === box || target === img || target.closest('[data-lightbox-close]')) {
      closeBox();
      return;
    }
    if (target.tagName !== 'IMG') return;
    var src = target.getAttribute('src') || '';
    if (src.indexOf('/media?path=') === -1) return;
    img.setAttribute('src', src);
    img.setAttribute('alt', target.getAttribute('alt') || '');
    box.hidden = false;
    box.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lightbox-open');
  });
  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && !box.hidden) closeBox();
  });
}

studioRestoreModelPreferences();
studioInitReferencePickers();
studioInitProjectDeleteDialog();
studioInitBibleDetailDialogs();
studioInitLightbox();
studioRestoreScroll();
studioStartProjectPolling();
"""


CSS = """
/* Studio Agent — production console theme. Editing-bay dark palette, film-strip
   stage tracker as the signature element, monospace reserved for the data layer. */
:root {
  color-scheme: dark;
  --ink:#0E0F12; --surface:#16181D; --surface-2:#1D2027; --surface-3:#121419;
  --line:#2A2E37; --line-soft:#23272F;
  --text:#ECEEF2; --muted:#98A0AE; --faint:#6A7280;
  --amber:#F2A33C; --amber-ink:#1A1206;
  --pass:#57B894; --fail:#E5645C; --run:#5BB8D6;
  --warn:#E8B33A; --info:var(--run);
  --r:6px; --r-lg:10px;
  --s1:4px; --s2:8px; --s3:12px; --s4:16px; --s5:24px; --s6:32px;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing:border-box; }
html { -webkit-text-size-adjust:100%; }
body { margin:0; background:var(--ink); color:var(--text); font:14px/1.5 var(--sans); }
/* SMPTE color-bar signature strip across the very top of every page */
body::before { content:''; display:block; height:3px; background:linear-gradient(90deg,
  #C9C9C9 0 14.28%, #D4D44A 0 28.56%, #4ED1D1 0 42.84%, #4ED14E 0 57.12%,
  #D14ED1 0 71.4%, #D14E4E 0 85.68%, #4E6AD1 0 100%); }

a { color:var(--amber); text-decoration:none; }
a:hover { text-decoration:underline; }
:focus-visible { outline:2px solid var(--amber); outline-offset:2px; border-radius:3px; }

h1 { margin:0; font-size:18px; font-weight:650; letter-spacing:.2px; }
h2 { margin:0 0 var(--s3); font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
h3 { margin:0 0 var(--s2); font-size:13px; font-weight:650; }
h4 { margin:var(--s3) 0 var(--s1); font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:700; }
p { margin:var(--s2) 0; }
.muted { color:var(--muted); margin:0; }
.mono, code { font-family:var(--mono); }

/* Top bar */
.toolbar { display:flex; align-items:center; gap:var(--s4); padding:var(--s3) var(--s5); border-bottom:1px solid var(--line); background:var(--surface); }
.toolbar .back { color:var(--muted); font-size:12px; font-weight:600; }
.toolbar .back:hover { color:var(--text); text-decoration:none; }
.toolbar .brand { display:inline-flex; align-items:center; gap:7px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; font-size:12px; color:var(--text); }
.toolbar .brand::before { content:'◆'; color:var(--amber); }
.toolbar .eyebrow { font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.toolbar .crumb { font-family:var(--mono); font-size:13px; color:var(--amber); letter-spacing:-.01em; }
h2 { letter-spacing:.16em; }
.toolbar .sub { margin-left:auto; color:var(--muted); max-width:48%; text-align:right; overflow-wrap:anywhere; }
.lang-toggle { display:inline-flex; align-items:center; gap:2px; padding:3px; border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); }
.lang-toggle button { min-height:28px; padding:4px 9px; border:0; background:transparent; color:var(--muted); font-size:11px; line-height:1; }
.lang-toggle button.active { background:var(--amber); color:var(--amber-ink); }

main { padding:var(--s5); max-width:1280px; margin:0 auto; }
.dashboard, .stack { display:grid; grid-template-columns:1fr; gap:var(--s4); }
section { border:1px solid var(--line); background:var(--surface); border-radius:var(--r-lg); padding:var(--s4); }
section[id] { scroll-margin-top:64px; }
.section-head { display:flex; align-items:baseline; justify-content:space-between; gap:var(--s3); margin-bottom:var(--s3); }
.section-head h2 { margin:0; }

/* Tables */
table { width:100%; border-collapse:collapse; }
th { text-align:left; font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-weight:700; padding:var(--s2); border-bottom:1px solid var(--line); }
td { padding:var(--s3) var(--s2); border-bottom:1px solid var(--line-soft); vertical-align:top; overflow-wrap:anywhere; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:var(--surface-2); }
.table-scroll { overflow-x:auto; }
.project-actions { width:1%; white-space:nowrap; }

/* Forms & buttons */
label { display:grid; gap:var(--s1); color:var(--muted); font-size:12px; }
button, input, select, textarea { font:inherit; color:var(--text); background:var(--surface-3); border:1px solid var(--line); border-radius:var(--r); padding:8px 10px; min-height:36px; }
input::placeholder, textarea::placeholder { color:var(--faint); }
select, textarea { width:100%; }
textarea { min-height:96px; resize:vertical; line-height:1.5; }
textarea.artifact-editor { min-height:68vh; font:13px/1.5 var(--mono); }
button { cursor:pointer; font-weight:600; background:var(--surface-2); border-color:var(--line); }
button:hover { border-color:var(--faint); }
button.primary { background:var(--amber); border-color:var(--amber); color:var(--amber-ink); font-weight:700; }
button.primary:hover { filter:brightness(1.06); border-color:var(--amber); }
button.secondary { background:var(--surface-2); color:var(--text); }
button.danger { background:transparent; border-color:var(--warn); color:var(--warn); }
button.danger:hover { background:rgba(232,179,58,.12); border-color:var(--warn); }
button.delete-trigger { min-height:32px; padding:5px 10px; background:transparent; border-color:rgba(255,92,92,.48); color:var(--fail); }
button.delete-trigger:hover { background:rgba(255,92,92,.1); border-color:var(--fail); }
button.delete-confirm { background:var(--fail); border-color:var(--fail); color:#160707; font-weight:750; }
button.delete-confirm:hover { filter:brightness(1.08); border-color:var(--fail); }
body.is-busy { cursor:wait; }
body.is-busy::after { content:''; position:fixed; z-index:10000; top:3px; left:0; height:4px; width:42%; background:linear-gradient(90deg, transparent, var(--amber), var(--info), transparent); border-radius:999px; animation:busy-progress 1.1s ease-in-out infinite; box-shadow:0 0 18px rgba(245,166,35,.55); }
button[aria-busy="true"] { opacity:.85; cursor:wait; }
.busy-note { margin:0; padding:8px 10px; border:1px solid rgba(245,166,35,.35); border-radius:var(--r); background:rgba(245,166,35,.1); color:var(--amber); font-size:12px; font-weight:650; display:flex; align-items:center; gap:8px; }
.busy-note::before { content:''; width:12px; height:12px; border:2px solid rgba(245,166,35,.25); border-top-color:var(--amber); border-radius:50%; animation:busy-spin .8s linear infinite; flex:0 0 auto; }
.refresh-notice { position:fixed; right:var(--s5); bottom:var(--s5); z-index:10001; display:flex; align-items:center; gap:var(--s3); max-width:min(420px, calc(100vw - 48px)); padding:12px 14px; border:1px solid rgba(91,200,230,.45); border-radius:var(--r-lg); background:rgba(17,20,26,.96); box-shadow:0 16px 48px rgba(0,0,0,.35); color:var(--text); }
.refresh-notice span { color:var(--text); }
.refresh-notice button { min-height:32px; padding:6px 10px; border-color:rgba(91,200,230,.5); color:var(--info); white-space:nowrap; }
@keyframes busy-progress { 0% { transform:translateX(-45vw); } 50% { transform:translateX(70vw); } 100% { transform:translateX(115vw); } }
@keyframes stage-progress { 0% { transform:translateX(0); } 100% { transform:translateX(390%); } }
@keyframes busy-spin { to { transform:rotate(360deg); } }

/* Run panel call sheet */
.run-panel { display:grid; grid-template-columns:minmax(220px,.7fr) minmax(320px,1.3fr); gap:var(--s5); align-items:start; }
.run-form { display:grid; grid-template-columns:repeat(4, minmax(150px,1fr)); gap:var(--s3); }
.run-form .full { grid-column:1 / -1; }
.model-mix { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-2); padding:var(--s3); display:grid; gap:var(--s3); }
/* label/.model-mix set display, which beats the UA's [hidden] rule — restate it for the
   mode-dependent clip-plan block and Length field so the format toggle really hides them. */
[data-clip-plan][hidden], [data-length-label][hidden] { display:none; }
.model-mix h3 { margin:0; font-size:13px; }
.model-mix-grid { display:grid; grid-template-columns:repeat(4, minmax(150px,1fr)); gap:var(--s3); }
.run-mode { display:flex; flex-wrap:wrap; gap:var(--s2); }
.run-mode label { flex-direction:row; align-items:center; gap:var(--s2); min-height:38px; padding:8px 12px; border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); color:var(--text); cursor:pointer; }
.run-mode input { min-height:auto; }
.run-actions { display:flex; flex-wrap:wrap; gap:var(--s2); }

/* Reference contact sheet */
.reference-picker { display:grid; gap:var(--s2); min-width:0; }
.reference-picker-hint { margin:0; color:var(--faint); font-size:11px; }
.reference-preview[hidden] { display:none; }
.reference-preview { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px,1fr)); gap:var(--s2); }
.reference-preview-card { position:relative; min-width:0; overflow:hidden; border:1px solid var(--line); border-radius:var(--r); background:var(--surface-2); box-shadow:0 8px 22px rgba(0,0,0,.18); }
.reference-preview-card img { width:100%; height:112px; max-height:112px; object-fit:cover; border:0; border-bottom:1px solid var(--line); border-radius:0; }
.reference-remove { position:absolute; z-index:1; top:6px; right:6px; width:30px; min-height:30px; padding:0; border-radius:50%; border-color:rgba(255,255,255,.35); background:rgba(15,17,21,.9); color:#fff; font-size:20px; line-height:1; box-shadow:0 3px 12px rgba(0,0,0,.45); }
.reference-remove:hover { background:var(--fail); border-color:var(--fail); }
.reference-preview-meta { display:grid; gap:5px; padding:8px; }
.reference-alias { justify-self:start; min-height:28px; padding:3px 8px; border-color:rgba(245,166,35,.5); background:rgba(245,166,35,.12); color:var(--amber); font:700 11px/1.2 var(--mono); }
.reference-alias:hover { border-color:var(--amber); background:rgba(245,166,35,.2); }
.reference-filename { min-width:0; overflow:hidden; color:var(--muted); font:10px/1.35 var(--mono); text-overflow:ellipsis; white-space:nowrap; }

/* Job monitor */
.jobs { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px,1fr)); gap:var(--s3); }
.job { border:1px solid var(--line); background:var(--surface-2); border-radius:var(--r); padding:var(--s3); display:grid; gap:var(--s2); }
.job header { display:flex; align-items:center; justify-content:space-between; gap:var(--s2); }
.job p { margin:0; overflow-wrap:anywhere; }
.job dl { display:grid; grid-template-columns:64px 1fr; gap:var(--s1) var(--s2); margin:0; }
.job dt { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
.job dd { margin:0; color:var(--text); font-family:var(--mono); font-size:12px; overflow-wrap:anywhere; }
.error { color:var(--fail); margin:0; }

/* Console header + film-strip stage tracker (signature) */
.console { display:grid; gap:var(--s4); }
.filmstrip { list-style:none; margin:0; padding:10px 0; display:flex; gap:6px; overflow-x:auto; }
.filmstrip .frame { position:relative; flex:1 0 96px; min-width:96px; border:1px solid var(--line);
  border-radius:var(--r); background:var(--surface-3); padding:16px 10px 14px; display:grid; gap:3px;
  color:var(--text); text-decoration:none; transition:border-color .15s, background .15s; }
.filmstrip .frame::before, .filmstrip .frame::after { content:''; position:absolute; left:8px; right:8px;
  height:3px; background-image:repeating-linear-gradient(90deg, var(--line) 0 5px, transparent 5px 11px);
  opacity:.7; }
.filmstrip .frame::before { top:4px; } .filmstrip .frame::after { bottom:4px; }
.filmstrip .frame .no { font-family:var(--mono); font-size:10px; color:var(--faint); }
.filmstrip .frame .nm { font-size:12px; font-weight:650; text-transform:capitalize; }
.filmstrip .frame .st { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.filmstrip .frame:hover { border-color:var(--faint); text-decoration:none; }
.frame.latent { opacity:.62; }
.frame.done .no { color:var(--pass); }
.frame.done .nm::after { content:' ✓'; color:var(--pass); }
.frame.active { border-color:var(--amber);
  background:linear-gradient(180deg, rgba(242,163,60,.14), transparent);
  box-shadow:inset 0 0 0 1px var(--amber), 0 0 22px rgba(242,163,60,.18); }
.frame.active .nm, .frame.active .no { color:var(--amber); }
.frame.active::after { height:6px; bottom:-9px; left:50%; width:10px;
  transform:translateX(-50%); border-radius:2px; background:var(--amber); opacity:.9; }
.frame.selected { outline:2px solid var(--amber); outline-offset:2px; }
.filmstrip .frame:focus-visible { outline:2px solid var(--run); outline-offset:2px; }

.stats { display:flex; flex-wrap:wrap; gap:var(--s6); }
.stat { display:grid; gap:2px; }
.stat .k { font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.stat .v { font-family:var(--mono); font-size:15px; color:var(--text); }
.controls { display:flex; flex-wrap:wrap; gap:var(--s2); align-items:center; }
.controls form { display:inline-flex; gap:var(--s2); margin:0; }

/* Media */
video, img { display:block; width:100%; max-height:72vh; object-fit:contain; background:#000; border:1px solid var(--line); border-radius:var(--r); }
audio { width:100%; }
pre { max-height:360px; overflow:auto; margin:0; padding:var(--s3); white-space:pre-wrap; background:var(--surface-3); border:1px solid var(--line-soft); border-radius:var(--r); color:var(--muted); font:12px/1.5 var(--mono); }

.notice { margin:var(--s4) var(--s5) 0; padding:10px 14px; border:1px solid var(--amber); border-left-width:3px; border-radius:var(--r); background:rgba(245,166,35,.1); color:var(--text); }

/* Project deletion confirmation */
dialog.delete-dialog { width:min(460px, calc(100vw - 32px)); padding:0; color:var(--text); border:1px solid rgba(255,92,92,.55); border-radius:var(--r-lg); background:var(--surface); box-shadow:0 28px 90px rgba(0,0,0,.68), 0 0 0 1px rgba(255,92,92,.08); }
dialog.delete-dialog[open] { animation:delete-dialog-in .16s ease-out; }
dialog.delete-dialog::backdrop { background:rgba(4,5,8,.78); backdrop-filter:blur(3px); }
.delete-dialog form { display:grid; grid-template-columns:42px 1fr; gap:var(--s3); margin:0; padding:var(--s5); }
.delete-dialog-mark { display:grid; place-items:center; width:42px; height:42px; border:1px solid rgba(255,92,92,.5); border-radius:50%; background:rgba(255,92,92,.12); color:var(--fail); font:800 20px/1 var(--mono); }
.delete-dialog-kicker { display:block; margin-bottom:3px; color:var(--fail); font:700 10px/1.2 var(--mono); letter-spacing:.12em; text-transform:uppercase; }
.delete-dialog h3 { margin:0; font-size:18px; }
.delete-dialog-warning, .delete-dialog-project, .delete-dialog .muted, .delete-dialog-actions { grid-column:1 / -1; }
.delete-dialog-warning { margin:var(--s2) 0 0; color:var(--text); }
.delete-dialog-project { margin:0; padding:10px 12px; overflow-wrap:anywhere; border:1px solid rgba(255,92,92,.3); border-radius:var(--r); background:var(--surface-3); color:var(--fail); }
.delete-dialog-actions { display:flex; justify-content:flex-end; gap:var(--s2); margin-top:var(--s2); }
@keyframes delete-dialog-in { from { opacity:0; transform:translateY(8px) scale(.985); } to { opacity:1; transform:none; } }

/* Focused Stage Workbench */
.stage-workbench { padding:0; overflow:hidden; border-color:rgba(245,166,35,.35); }
.stage-workbench > .section-head { padding:var(--s4); margin:0; border-bottom:1px solid var(--line); background:linear-gradient(180deg, rgba(245,166,35,.08), transparent); }
.stage-workbench h3 { font-size:18px; margin:2px 0 4px; }
.stage-workbench.is-loading { box-shadow:0 0 0 1px rgba(245,166,35,.45), 0 0 36px rgba(245,166,35,.08); }
.stage-loading { position:relative; display:flex; align-items:center; gap:var(--s3); padding:12px var(--s4); border-bottom:1px solid rgba(245,166,35,.25); background:rgba(245,166,35,.08); overflow:hidden; }
.stage-loading::after { content:''; position:absolute; left:-35%; bottom:0; width:35%; height:3px; background:linear-gradient(90deg, transparent, var(--amber), var(--info), transparent); animation:stage-progress 1.15s ease-in-out infinite; }
.stage-loading b { display:block; margin-bottom:2px; color:var(--text); }
.stage-spinner { width:18px; height:18px; border:2px solid rgba(245,166,35,.25); border-top-color:var(--amber); border-radius:50%; animation:busy-spin .8s linear infinite; flex:0 0 auto; }
.workbench-layout { display:grid; grid-template-columns:minmax(0,1fr) 340px; min-height:420px; }
.workbench-main { padding:var(--s4); overflow:auto; max-height:980px; }
.workbench-side { display:grid; align-content:start; gap:var(--s3); padding:var(--s4); border-left:1px solid var(--line); background:var(--surface-2); }
.workbench-box { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); padding:var(--s3); display:grid; gap:var(--s3); }
.workbench-box form { display:grid; gap:var(--s3); margin:0; }
.workbench-readable { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); padding:var(--s4); }
.workbench-readable ul { margin:0; padding-left:18px; }
.workbench-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:var(--s3); }
.workbench-card { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); padding:var(--s3); display:grid; gap:var(--s2); }
.workbench-card header { display:flex; align-items:baseline; justify-content:space-between; gap:var(--s2); color:var(--muted); font-size:11px; }
.workbench-card header b { color:var(--text); font-size:13px; }
.workbench-card p { margin:0; }
.workbench-card ul { margin:0; padding-left:18px; }
.workbench-card img, .workbench-card video { max-height:260px; }
.prompt-workbench-wrap { min-width:0; }
.prompt-review-layout { display:grid; grid-template-columns:230px minmax(0,1fr) 330px; min-height:560px; }
.prompt-shot-rail, .prompt-conversation { padding:var(--s4); background:var(--surface-2); display:grid; align-content:start; gap:var(--s3); }
.prompt-shot-rail { border-right:1px solid var(--line); }
.prompt-conversation { border-left:1px solid var(--line); }
.prompt-shot { display:grid; gap:2px; padding:10px; border:1px solid var(--line); border-radius:var(--r); color:var(--text); text-decoration:none; }
.prompt-shot.ready { border-color:rgba(89,196,132,.45); }
.prompt-shot.invalid, .prompt-errors { border-color:var(--warn); color:var(--warn); }
.prompt-blockers { border:1px solid var(--warn); border-radius:var(--r); padding:var(--s3); color:var(--warn); font-size:.9em; }
.prompt-blockers ul { margin:6px 0 0; padding-left:1.2em; }
.prompt-blockers a { color:var(--warn); font-weight:650; }
.prompt-review-main { min-width:0; padding:var(--s4); display:grid; align-content:start; gap:var(--s3); }
.prompt-review-main > header { display:flex; justify-content:space-between; gap:var(--s3); align-items:baseline; }
.exact-prompt { white-space:pre-wrap; overflow-wrap:anywhere; max-height:520px; overflow:auto; padding:var(--s3); border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); }
.prompt-keyframe { max-height:300px; width:100%; object-fit:contain; background:#090b0e; }
.panel-plan { margin:0; padding-left:22px; }
.conversation-turn { padding:10px; border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); }
.conversation-turn p { margin:4px 0 0; white-space:pre-wrap; }
.conversation-turn.user { border-color:rgba(91,200,230,.4); }
.prompt-conversation form { display:grid; gap:var(--s3); }
.keyframe-feedback { display:grid; gap:var(--s2); padding-top:var(--s2); border-top:1px solid var(--line-soft); }
.keyframe-feedback textarea { min-height:88px; }
.shot-structure { display:grid; gap:var(--s2); padding-top:var(--s2); border-top:1px solid var(--line-soft); }
.shot-structure .clip-duration { display:flex; align-items:end; gap:var(--s2); }
.add-shot summary { cursor:pointer; color:var(--muted); font-size:12px; }
.add-shot form { display:grid; gap:var(--s2); padding-top:var(--s2); }
.add-shot textarea { min-height:66px; }
.workbench-media-row { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; min-height:0; }
.workbench-media-row:empty { display:none; }
.workbench-media-row img { max-height:140px; }
.bible-image { min-width:0; margin:0; display:grid; gap:4px; }
.bible-image figcaption { color:var(--muted); font:10px/1.3 var(--mono); letter-spacing:.04em; text-align:center; text-transform:uppercase; }
.bible-card-controls { margin-top:12px; padding-top:12px; border-top:1px solid var(--line); display:grid; gap:10px; }
.bible-card-controls textarea { width:100%; min-height:54px; }
.bible-controls { display:grid; gap:6px; }
.bible-banner { color:var(--warn, #d08700); font:12px/1.4 var(--mono); margin:0; }
.asset-actions { display:grid; gap:var(--s2); }

.studio-subsection { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-3); padding:var(--s4); }
.studio-subsection p { margin:6px 0; }
.studio-cards { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px,1fr)); gap:var(--s3); }
.studio-card { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-2); padding:var(--s3); display:grid; gap:7px; }
.studio-card header { display:flex; justify-content:space-between; gap:var(--s2); color:var(--muted); font-size:11px; }
.studio-card header b { color:var(--text); font-size:13px; }
.studio-card p { margin:0; }
.studio-card ul, .history ul { margin:0; padding-left:18px; }
img[src*="/media?path="] { cursor:zoom-in; }
.lightbox { position:fixed; inset:0; z-index:1000; display:flex; align-items:center; justify-content:center; padding:32px; background:rgba(6,8,12,.92); cursor:zoom-out; }
.lightbox[hidden] { display:none; }
.lightbox img { max-width:96vw; max-height:92vh; width:auto; height:auto; border-radius:6px; box-shadow:0 12px 48px rgba(0,0,0,.6); cursor:default; }
.lightbox-close { position:fixed; top:18px; right:24px; width:40px; height:40px; font-size:26px; line-height:1; color:var(--text); background:var(--surface-2); border:1px solid var(--line); border-radius:999px; cursor:pointer; }
body.lightbox-open { overflow:hidden; }
.identity-board-block { border-top:1px solid var(--line); padding-top:8px; }
.identity-board-block h4 { margin:0 0 6px; font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); }
dl.identity-board { display:grid; gap:4px; margin:0; }
dl.identity-board .id-row { display:grid; grid-template-columns:78px 1fr; gap:8px; align-items:baseline; }
dl.identity-board dt { margin:0; font-size:11px; color:var(--muted); }
dl.identity-board dd { margin:0; }
.id-rules { display:grid; grid-template-columns:1fr 1fr; gap:var(--s3); margin-top:8px; }
.id-rule h5 { margin:0 0 2px; font-size:11px; letter-spacing:.04em; text-transform:uppercase; }
.id-rule.do h5 { color:var(--pass); }
.id-rule.dont h5 { color:var(--fail); }
.id-rule ul { margin:0; padding-left:16px; }
.id-points { margin-top:8px; }
.id-points h5 { margin:0 0 2px; font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }
.id-points ul { margin:0; padding-left:16px; display:grid; gap:2px; }
.palette-row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.palette-chips { display:inline-flex; align-items:center; gap:6px; flex-wrap:wrap; }
.palette-chips .swatch { width:14px; height:14px; border-radius:3px; border:1px solid var(--line); display:inline-block; }
.palette-chips code { font-size:11px; color:var(--muted); margin-right:4px; }
.reference-upload { display:grid; grid-template-columns:1fr 1fr; gap:var(--s3); margin-bottom:var(--s3); }
.reference-upload .full { grid-column:1 / -1; }
.reference-card img { height:160px; max-height:160px; }
.history { border-top:1px solid var(--line); padding-top:var(--s3); }

/* Shot review cards */
.shots-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px,1fr)); gap:var(--s4); }
.shot { border:1px solid var(--line); border-radius:var(--r); background:var(--surface-2); padding:var(--s3); display:flex; flex-direction:column; gap:var(--s3); }
.shot header { display:flex; align-items:center; justify-content:space-between; gap:var(--s2); font-family:var(--mono); }
.shot video, .shot img { max-height:none; }
.shot .muted { margin:0; }

/* Badges */
.badge { font-family:var(--mono); font-size:10px; letter-spacing:.06em; padding:3px 8px; border:1px solid var(--line); border-radius:999px; color:var(--muted); text-transform:uppercase; white-space:nowrap; }
.badge.pass, .badge.done { color:var(--pass); border-color:rgba(78,216,138,.4); background:rgba(78,216,138,.1); }
.badge.warn, .badge.paused { color:var(--warn); border-color:rgba(232,179,58,.4); background:rgba(232,179,58,.1); }
.badge.fail, .badge.failed { color:var(--fail); border-color:rgba(255,92,92,.4); background:rgba(255,92,92,.1); }
.badge.running { color:var(--info); border-color:rgba(91,200,230,.4); background:rgba(91,200,230,.1); }
.badge.none, .badge.queued { color:var(--faint); }
ul.fails { margin:6px 0 0; padding-left:18px; color:var(--warn); }
ul.fails b { color:var(--text); }

.all-files { border:1px solid var(--line); background:var(--surface); border-radius:var(--r-lg); padding:var(--s4); }
.all-files > summary { cursor:pointer; font-size:11px; font-weight:700; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); }
.all-files[open] > summary { margin-bottom:var(--s3); }
.workbench-main { animation:bay-scrub .18s ease-out; }
@keyframes bay-scrub { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) {
  .workbench-main { animation:none; }
  .filmstrip .frame { transition:none; }
  body.is-busy::after { animation:none; }
  .stage-loading::after { animation:none; }
  .stage-spinner { animation:none; }
  .busy-note::before { animation:none; }
}
@media (max-width: 1100px) { .workbench-layout, .prompt-review-layout { grid-template-columns:1fr; } .workbench-side, .prompt-conversation { border-left:0; border-top:1px solid var(--line); } .prompt-shot-rail { border-right:0; border-bottom:1px solid var(--line); } }
@media (max-width: 920px) { .run-panel, .run-form, .model-mix-grid, .reference-upload { grid-template-columns:1fr; } .toolbar, .section-head { align-items:flex-start; flex-direction:column; } .toolbar .sub { margin-left:0; text-align:left; max-width:100%; } .reference-upload .full { grid-column:auto; } }

/* Bible workbench: large single-column cards + read-only detail popup */
.bible-grid { grid-template-columns: 1fr; gap: 18px; }
.bible-card { display: flex; flex-direction: column; gap: 14px; }
.bible-media-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 12px; }
.bible-media-row .bible-image:first-child img { max-width: 520px; width: 100%; height: auto; border-radius: 10px; }
.bible-media-row .bible-image:not(:first-child) img { max-width: 120px; height: auto; border-radius: 8px; opacity: .92; }
.bible-card-meta { display: flex; align-items: center; gap: 12px; }
.bible-details-link {
  background: none; border: 1px solid var(--line, #3a3f4b); color: inherit;
  border-radius: 999px; padding: 6px 14px; cursor: pointer; font: inherit;
}
.bible-details-link:hover { border-color: var(--accent, #7aa2f7); }
.bible-detail-dialog {
  border: 1px solid var(--line, #3a3f4b); border-radius: 14px; padding: 0;
  max-width: 640px; width: calc(100% - 32px); max-height: 80vh;
  background: var(--panel, #1b1e27); color: inherit;
}
.bible-detail-dialog::backdrop { background: rgba(0,0,0,.55); }
.bible-detail-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 16px 20px; border-bottom: 1px solid var(--line, #3a3f4b); margin: 0;
}
.bible-detail-head h3 { margin: 0; }
.bible-detail-close {
  background: none; border: none; color: inherit; font-size: 22px;
  line-height: 1; cursor: pointer; padding: 0 4px;
}
.bible-detail-body { padding: 16px 20px; overflow: auto; max-height: 56vh; }
.bible-detail-body section { margin-bottom: 16px; }
.bible-detail-foot { padding: 12px 20px; border-top: 1px solid var(--line, #3a3f4b); }
"""
