"""Tests for the dependency-free local web app helpers."""

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from studio_agent.orchestrator.project import Project
from studio_agent.reference_assets import save_reference_intake_uploads
from studio_agent.web import (
    BUSY_JS,
    JobRunner,
    ProjectBusyError,
    RunJob,
    UploadedFormFile,
    _compose_model_mix,
    delete_project,
    make_handler,
    project_artifacts,
    project_state,
    project_summaries,
    render_index,
    render_project,
    safe_project_path,
    shot_reviews,
)


def _project_with_shot(tmp_path, *, qc):
    p = Project.create("review", root=tmp_path, stages=["video", "review"])
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "duration_s": 3.0}
    ]}))
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("assets", "qc", "sh-001.json").write_text(json.dumps(qc))
    return p


def _full_pipeline_project(tmp_path):
    p = Project.create(
        "full pipeline", root=tmp_path,
        stages=["plot", "script", "bible", "storyboard",
                "video", "review", "audio", "assemble"],
    )
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A memory thief learns restraint.",
        "synopsis": "The thief must choose between power and connection.",
        "themes": ["power"],
    }))
    p.path("story", "script.json").write_text(json.dumps({
        "episodes": [{"scenes": [{
            "scene": 1, "heading": "INT. SCHOOL - DAY",
            "beats": ["A student hides a new power."],
            "dialogue": [{"character": "Mara", "line": "Not yet."}],
        }]}],
    }))
    ldir = p.path("bible", "locations", "school")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": "School Hallway", "description": "A fluorescent school corridor.",
        "palette": "green lockers", "lighting": "overhead fluorescent",
    }))
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001", "scene": 1, "camera": "close-up",
        "action": "A student clenches a glowing fist.",
        "duration_s": 3, "keyframe": "sh-001.png", "characters": ["Mara"],
    }]}))
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    p.path("assets", "qc", "sh-001.json").write_text(json.dumps({
        "overall_pass": True, "summary": "approved"}))
    p.path("assets", "qc", "summary.json").write_text(json.dumps({"all_passed": True}))
    return p


def test_render_project_selects_requested_stage_and_defaults_to_gate(tmp_path):
    from studio_agent.web import render_project
    p = _full_pipeline_project(tmp_path)
    default_html = render_project(p)
    script_html = render_project(p, stage="script")
    bogus_html = render_project(p, stage="not-a-stage")

    # Rail frames link to ?stage=<name> for every stage.
    for stage in p.stages:
        assert f"?stage={stage}" in default_html

    def selected_stage(html):
        m = re.search(r"frame[^']*selected' href='[^']*\?stage=([^']+)'", html)
        return m.group(1) if m else None

    gate = p.current_stage or p.stages[-1]
    assert selected_stage(default_html) == gate     # default selection = active gate
    assert selected_stage(script_html) == "script"  # explicit selection wins
    assert selected_stage(bogus_html) == gate       # unknown stage falls back to gate


def test_project_writer_rejects_second_writer_and_releases_after_exit():
    jobs = JobRunner(inline=False)

    with jobs.project_writer("film-1"):
        assert jobs.has_active_project_job("film-1") is True
        with pytest.raises(ProjectBusyError, match="active job"):
            with jobs.project_writer("film-1"):
                pass

    assert jobs.has_active_project_job("film-1") is False
    with jobs.project_writer("film-1"):
        assert jobs.has_active_project_job("film-1") is True


def test_start_resume_rejects_project_with_active_writer(tmp_path):
    jobs = JobRunner(inline=False)

    with jobs.project_writer("film-1"):
        with pytest.raises(ProjectBusyError, match="active job"):
            jobs.start_resume(tmp_path, project_id="film-1")


def test_http_mutation_returns_conflict_while_project_writer_is_active(tmp_path):
    project = Project.create("busy http mutation", root=tmp_path, stages=["plot"])
    project.set_stage_status("plot", "complete")
    project.current_stage = "plot"
    project.save()
    jobs = JobRunner(inline=False)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    try:
        with jobs.project_writer(project.project_id):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/projects/{project.project_id}/approve",
                data=b"",
                method="POST",
            )
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(request, timeout=30)
            body = exc_info.value.read().decode("utf-8")
            assert exc_info.value.code == 409
            assert "active job" in body
    finally:
        server.shutdown()
        thread.join(timeout=30)

    unchanged = Project.load(project.dir)
    assert unchanged.stage_status("plot") == "complete"
    assert unchanged.current_stage == "plot"


def test_same_idea_run_reports_busy_when_existing_project_has_writer(tmp_path):
    project = Project.create(
        "same busy idea",
        root=tmp_path,
        stages=["plot"],
        model_config={"llm": "fake"},
    )
    jobs = JobRunner(inline=True)

    with jobs.project_writer(project.project_id):
        job = jobs.start_run(
            tmp_path,
            idea="same busy idea",
            profile_name="fake",
            style_name="cinematic",
            format_name="short_film",
            language="en",
            auto=False,
        )

    assert job.status == "failed"
    assert "active job" in job.error


_FAILING_QC = {
    "overall_pass": False,
    "max_severity": "high",
    "recommendation": "regenerate",
    "checks": [
        {"dimension": "identity_drift", "passed": False, "severity": "high",
         "detail": "face morphs", "timestamp": "00:02"},
        {"dimension": "artifacts", "passed": True, "severity": "none", "detail": "ok"},
    ],
}

PNG_BYTES = b"\x89PNG\r\n\x1a\nweb-upload"


def test_project_summaries_read_file_backed_projects(tmp_path):
    p = Project.create(
        "a lighthouse short",
        root=tmp_path,
        stages=["plot"],
        model_config={"format_name": "short_film", "language": "en"},
    )
    p.add_cost(stage="plot", provider="fake", cost_usd=0.25, seconds=1.0)

    summaries = project_summaries(tmp_path)

    assert summaries == [{
        "id": p.project_id,
        "idea": "a lighthouse short",
        "status": "in_progress",
        "current_stage": "plot",
        "cost": 0.25,
        "profile": None,
        "format": "short_film",
        "language": "en",
    }]


def test_render_index_exposes_local_dashboard_controls(tmp_path):
    html = render_index(tmp_path)

    assert "New Run" in html
    assert "Review each stage" in html
    assert "Auto approve" in html
    # The "Model preset" dropdown is gone; model choice is the per-component mix.
    assert "Model preset" not in html
    assert "selected_profile" not in html
    assert "Model selection" in html
    assert "Story / script LLM" in html
    assert "Image / keyframes" in html
    assert "Video generation" in html
    # Automated LLM QC was removed, so there is no "Video QC" selector.
    assert "Video QC" not in html
    # New run buttons, all driven by the per-component selectors above.
    assert "Run full pipeline" in html
    assert "Cheap test run" not in html
    assert "__cheap_test__" not in html
    assert "name='profile'" not in html
    assert "Chinese Video Test" not in html
    assert "Run DeepSeek Story Test" not in html
    assert "Reference photos" in html
    assert "What are these for?" in html
    assert "data-reference-picker" in html
    assert "data-reference-preview" in html
    assert "data-reference-note='reference_note'" in html
    assert "data-alias-start='1'" in html
    assert "studioInitReferencePickers" in html
    assert "studioInsertReferenceAlias" in html
    assert "new DataTransfer()" in html
    assert "selectedFiles.splice(index, 1)" in html
    assert "URL.revokeObjectURL" in html
    assert "reference-remove" in html
    assert "reference-alias" in html
    assert "deepseek-v4-flash-260425" in html
    assert "doubao-seedance-2-0-fast-260128" in html
    assert "OpenAI image-2" in html
    assert "Doubao Seedream 5.0" in html
    assert "Chinese Seedance Fast" in html
    assert "Chinese Seedance Mini" in html
    # Automated LLM QC was removed, so VLM/QC model options no longer render here.
    assert "Doubao Seed 2.0 Lite QC" not in html
    assert "Claude Opus 4.8 QC" not in html
    assert "Job Monitor" in html
    for field in ("tone", "visual_world", "camera_language", "light_texture", "hard_avoidances"):
        assert f"name='{field}'" in html
    assert "data-lang-toggle" in html
    # the language map is now server-injected (generated from the Python catalog),
    # not a hand-kept client-side STUDIO_TEXT_ZH dictionary
    assert "window.STUDIO_I18N=" in html


def test_project_page_shows_decision_and_director_default(tmp_path):
    p = Project.create("decision web", root=tmp_path, stages=["bible"])
    p.current_stage = "bible"
    path = p.path("story", "decisions", "bible.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": "bible",
        "question": "Should Mara feel carefully composed or visibly worn down?",
        "choices": [
            {"value": "carefully composed", "label": "Carefully composed"},
            {"value": "visibly worn down", "label": "Visibly worn down"},
        ],
        "default": "carefully composed",
        "resolution": None,
    }))
    packet = p.path("knowledge", "packets", "identity-mara.json")
    packet.parent.mkdir(parents=True, exist_ok=True)
    packet.write_text(json.dumps({
        "purpose": "identity",
        "target": "mara",
        "selected_entries": [{
            "title": "Stable silhouette",
            "reasons": ["consistency"],
            "source": {"path": "core/foundations.yaml"},
        }],
    }))

    html = render_project(p)

    assert "Should Mara feel carefully composed" in html
    assert "Use the director" in html
    assert "/answer-decision" in html
    assert "Stable silhouette" in html
    assert "Refresh knowledge" in html
    assert "/refresh-knowledge" in html


def test_decision_non_default_label_not_translated(tmp_path):
    """Non-default choice labels (dynamic content) must not be routed through translator."""
    p = Project.create("decision label test", root=tmp_path, stages=["bible"])
    p.current_stage = "bible"
    path = p.path("story", "decisions", "bible.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a custom label that is NOT a catalog key to verify it's not translated.
    path.write_text(json.dumps({
        "stage": "bible",
        "question": "What mood?",
        "choices": [
            {"value": "default_mood", "label": "Use the director"},  # default choice
            {"value": "custom_mood", "label": "Make it noir"},  # custom non-default label
        ],
        "default": "default_mood",
        "resolution": None,
    }))

    html = render_project(p, lang="zh")

    # The default choice button should show translated "Use the director" (mocked to pass through in test).
    assert "Use the director" in html
    # The non-default choice label "Make it noir" must appear verbatim, not translated.
    assert "Make it noir" in html


def test_http_answer_decision_persists_resolution(tmp_path):
    p = Project.create(
        "decision post",
        root=tmp_path,
        stages=["bible"],
        model_config={"llm": "fake", "image": "fake"},
    )
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "A room remembers its former occupant.",
        "synopsis": "A room changes with each memory.",
        "characters": [],
        "arc": [],
    }))
    path = p.path("story", "decisions", "bible.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stage": "bible",
        "question": "Should the room feel cared for or abandoned?",
        "choices": [
            {"value": "cared for", "label": "Cared for"},
            {"value": "abandoned", "label": "Abandoned"},
        ],
        "default": "cared for",
        "resolution": None,
    }))
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        body = urllib.parse.urlencode({"stage": "bible", "choice": "abandoned"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/answer-decision",
            data=body,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30).read()  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    resolved = json.loads(path.read_text())
    assert resolved["resolution"] == {
        "value": "abandoned",
        "source": "user",
        "locked": True,
    }


def test_job_runner_stores_creative_inputs(tmp_path):
    jobs = JobRunner(inline=True)
    job = jobs.start_run(
        tmp_path,
        idea="a deliberate camera test",
        profile_name="fake",
        style_name="cinematic",
        format_name="short_video",
        language="en",
        auto=False,
        length="30",
        creative_inputs={
            "tone": "tender but uneasy",
            "camera_language": "patient observation",
            "hard_avoidances": ["unmotivated orbit"],
        },
    )

    project = Project.load(tmp_path / job.project_id)
    assert project.model_config["creative_inputs"]["tone"] == "tender but uneasy"
    assert project.model_config["creative_inputs"]["hard_avoidances"] == ["unmotivated orbit"]
    assert project.model_config["clip_target_duration_s"] == 30


def test_dashboard_music_checkbox_threads_into_model_config(tmp_path):
    jobs = JobRunner(inline=True)
    job = jobs.start_run(
        tmp_path,
        idea="a quiet town wakes",
        profile_name="custom-model-mix",
        model_parts={"llm": "fake", "image": "fake", "video": "fake", "vlm": "fake"},
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
        music_enabled=True,
        music_mood="warm strings",
    )

    assert job.status == "done"
    project = Project.load(tmp_path / job.project_id)
    assert project.model_config["music_enabled"] is True
    assert project.model_config["music_mood"] == "warm strings"


def test_dashboard_format_dropdown_is_film_and_clip_only(tmp_path):
    html = render_index(tmp_path)

    assert ">Film<" in html
    assert ">Clip<" in html
    # The overlapping/dev-only presets are gone from the dropdown.
    assert "Short drama" not in html
    assert "Series episode" not in html
    assert "Smoke test" not in html


def test_dashboard_length_input_adapts_to_format(tmp_path):
    html = render_index(tmp_path)

    # A single adaptive length field replaces the fixed clip-duration dropdown.
    assert "name='length'" in html
    assert "name='clip_duration'" not in html
    # The page exposes which formats are clip mode so the unit can switch min/sec.
    assert "STUDIO_FORMAT_MODES" in html
    assert '"short_video": "clip"' in html


def test_shared_script_preserves_scroll_across_forms_and_reloads():
    assert "var STUDIO_SCROLL_KEY = 'studioScrollRestore:v1';" in BUSY_JS
    assert "function studioRememberScroll()" in BUSY_JS
    assert "function studioRestoreScroll()" in BUSY_JS
    assert "sessionStorage.setItem(STUDIO_SCROLL_KEY" in BUSY_JS
    assert "sessionStorage.removeItem(STUDIO_SCROLL_KEY)" in BUSY_JS
    assert "snapshot.path !== window.location.pathname" in BUSY_JS
    assert "Date.now() - Number(snapshot.savedAt || 0) > STUDIO_SCROLL_MAX_AGE_MS" in BUSY_JS
    assert "document.getElementById(snapshot.anchorId)" in BUSY_JS
    assert "document.addEventListener('submit', studioRememberScroll, true);" in BUSY_JS
    assert BUSY_JS.count("studioRememberScroll();\n      window.location.reload();") == 2
    assert "studioRestoreScroll();" in BUSY_JS


def test_render_index_marks_project_list_as_scroll_anchor(tmp_path):
    html = render_index(tmp_path)

    assert "<section id='projects' class='wide projects-panel'>" in html


def test_render_index_adds_delete_control_and_confirmation_dialog(tmp_path):
    project = Project.create("delete from dashboard", root=tmp_path, stages=["plot"])

    html = render_index(tmp_path)

    assert "<th>Actions</th>" in html
    assert "data-delete-project" in html
    assert f"data-project-id='{project.project_id}'" in html
    assert "data-delete-dialog" in html
    assert "Permanently delete project?" in html
    assert "Delete project" in html
    assert "Cancel" in html
    assert "question + '\\n\\n'" in BUSY_JS


def test_compose_model_mix_allows_cross_vendor_combinations():
    from studio_agent import cli

    profile = _compose_model_mix(cli.load_config(), {
        "llm": "volcengine-deepseek-v4-flash",
        "image": "openai-image-2",
        "video": "china-seedance-fast",
        "vlm": "gemini-cheap",
    })

    assert profile["llm"] == "openai-compatible"
    assert profile["llm_api_key_env"] == "ARK_API_KEY"
    assert profile["llm_model"] == "deepseek-v4-flash-260425"
    assert profile["image"] == "openai"
    assert profile["image_model"] == "gpt-image-2"
    assert profile["video"] == "volcengine"
    assert profile["video_model"] == "doubao-seedance-2-0-fast-260128"
    assert profile["vlm"] == "gemini"
    assert profile["vlm_model"] == "gemini-2.5-flash-lite"
    assert profile["tts"] == "fake"
    assert profile["music"] == "fake"


def test_compose_model_mix_reports_unknown_option():
    from studio_agent import cli

    with pytest.raises(ValueError, match="unknown video model option"):
        _compose_model_mix(cli.load_config(), {
            "llm": "fake",
            "image": "fake",
            "video": "missing-video",
            "vlm": "fake",
        })


def test_job_runner_launches_fake_project_inline(tmp_path):
    jobs = JobRunner(inline=True)

    job = jobs.start_run(
        tmp_path,
        idea="a dashboard smoke test",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
    )

    assert job.status == "done"
    assert job.project_id
    project = Project.load(tmp_path / job.project_id)
    assert project.model_config["profile"] == "fake"
    assert project.model_config["style_name"] == "anime"
    assert project.model_config["format_name"] == "short_film"
    assert project.model_config["audio_mode"] == "native_video"


def test_job_runner_launches_custom_model_mix_inline(tmp_path):
    jobs = JobRunner(inline=True)

    job = jobs.start_run(
        tmp_path,
        idea="a dashboard custom model mix test",
        profile_name="custom-model-mix",
        model_parts={
            "llm": "fake",
            "image": "fake",
            "video": "fake",
            "vlm": "fake",
        },
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
    )

    assert job.status == "done"
    project = Project.load(tmp_path / job.project_id)
    assert project.model_config["profile"] == "custom-model-mix"
    assert project.model_config["model_parts"] == {
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "vlm": "fake",
    }
    assert project.model_config["llm"] == "fake"
    assert project.model_config["image"] == "fake"
    assert project.model_config["video"] == "fake"
    assert project.model_config["audio_mode"] == "native_video"


def test_job_runner_saves_initial_reference_uploads_inline(tmp_path):
    jobs = JobRunner(inline=True)

    job = jobs.start_run(
        tmp_path,
        idea="a dashboard initial reference test",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=False,
        reference_uploads=[
            UploadedFormFile("hero.png", "image/png", PNG_BYTES),
            UploadedFormFile("apartment.png", "image/png", PNG_BYTES),
        ],
        reference_note="first image is protagonist; second image is the apartment",
    )

    project = Project.load(tmp_path / job.project_id)
    manifest = json.loads(project.path("references", "references.json").read_text())
    records = manifest["references"]
    assert [record["alias"] for record in records] == ["@image1", "@image2"]
    assert records[0]["pending_target"] == {"kind": "character_role", "value": "protagonist"}
    assert records[1]["pending_target"] == {"kind": "location", "value": "apartment"}
    assert project.model_config["vlm"] == "fake"


def test_job_runner_reports_failed_unknown_profile(tmp_path):
    jobs = JobRunner(inline=True)

    job = jobs.start_run(
        tmp_path,
        idea="a bad profile test",
        profile_name="missing-profile",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
    )

    assert job.status == "failed"
    assert "unknown profile" in job.error


def test_dashboard_style_description_creates_a_custom_style_project(tmp_path):
    jobs = JobRunner(inline=True)

    job = jobs.start_run(
        tmp_path,
        idea="a comet in the oven",
        profile_name="fake",
        style_description="1970s grainy analog sci-fi",
        format_name="short_film",
        language="en",
        auto=True,
    )

    assert job.status in {"paused", "done"}
    project = Project.load(tmp_path / job.project_id)
    assert project.model_config["style_name"] == "custom"
    assert project.model_config["style_input"]["description"] == "1970s grainy analog sci-fi"
    assert "sci-fi" in json.dumps(project.model_config["style"]).lower()


def test_dashboard_style_image_is_saved_as_a_style_reference(tmp_path):
    from studio_agent.reference_assets import style_reference_paths

    jobs = JobRunner(inline=True)
    job = jobs.start_run(
        tmp_path,
        idea="a detective in the neon rain",
        profile_name="fake",
        style_image=UploadedFormFile("moody-noir.png", "image/png", PNG_BYTES),
        format_name="short_film",
        language="en",
        auto=True,
    )

    assert job.status in {"paused", "done"}
    project = Project.load(tmp_path / job.project_id)
    # The dedicated style upload makes the project custom and is stored as a style/global ref.
    assert project.model_config["style_name"] == "custom"
    refs = style_reference_paths(project)
    assert refs and refs[0].endswith("moody-noir.png")


def test_job_runner_resumes_existing_project_in_background(tmp_path):
    from studio_agent import cli

    jobs = JobRunner(inline=True)
    # Review-mode run pauses at the first gate (plot), then we approve it.
    start = jobs.start_run(
        tmp_path,
        idea="a dashboard resume test",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=False,
    )
    project = Project.load(tmp_path / start.project_id)
    # The style stage runs first; review mode pauses at it. Approve it to reach plot.
    assert project.current_stage == "style"
    cli._machine().approve(project)
    assert project.current_stage == "plot"
    assert project.stage_status("plot") == "pending"

    resume = jobs.start_resume(tmp_path, project_id=project.project_id, auto=False)

    assert resume.status == "paused"
    assert resume.project_id == project.project_id
    advanced = Project.load(tmp_path / project.project_id)
    assert advanced.current_stage == "plot"
    assert advanced.stage_status("plot") == "complete"


def test_job_snapshot_includes_project_progress(tmp_path):
    jobs = JobRunner(inline=True)
    job = jobs.start_run(
        tmp_path,
        idea="a dashboard progress test",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
    )

    snapshot = jobs.snapshot(tmp_path)

    row = next(item for item in snapshot if item["id"] == job.id)
    assert row["status"] == "done"
    assert row["project_id"] == job.project_id
    assert row["project_status"] == "done"
    assert row["cost"] == 0.0


def test_http_dashboard_run_and_jobs_json(tmp_path):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "idea": "a local dashboard http run",
            "profile": "__custom__",
            "llm_option": "fake",
            "image_option": "fake",
            "video_option": "fake",
            "vlm_option": "fake",
            "style": "anime",
            "format": "short_film",
            "language": "en",
            "auto": "on",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/runs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert "Started custom-model-mix" in body
    assert status["jobs"][0]["status"] == "done"
    assert status["jobs"][0]["project_id"]
    assert "html" in status


def test_http_dashboard_custom_model_mix_and_jobs_json(tmp_path):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "idea": "a local dashboard custom model mix",
            "profile": "__custom__",
            "llm_option": "fake",
            "image_option": "fake",
            "video_option": "fake",
            "vlm_option": "fake",
            "style": "anime",
            "format": "short_film",
            "language": "en",
            "run_mode": "auto",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/runs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert "Started custom-model-mix" in body
    assert status["jobs"][0]["status"] == "done"
    assert status["jobs"][0]["profile"] == "custom-model-mix"
    assert status["jobs"][0]["model_parts"] == {
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "vlm": "fake",
    }
    assert "llm:fake" in status["html"]
    project = Project.load(tmp_path / status["jobs"][0]["project_id"])
    assert project.model_config["profile"] == "custom-model-mix"
    assert project.model_config["model_parts"]["video"] == "fake"


def test_http_dashboard_runs_exact_selected_model_mix(tmp_path):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "idea": "a local dashboard exact model mix run",
            "llm_option": "fake",
            "image_option": "fake",
            "video_option": "fake",
            "vlm_option": "fake",
            "style": "anime",
            "format": "short_film",
            "language": "en",
            "run_mode": "auto",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/runs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert status["jobs"][0]["status"] == "done"
    assert status["jobs"][0]["model_parts"] == {
        "llm": "fake",
        "image": "fake",
        "video": "fake",
        "vlm": "fake",
    }


def test_http_dashboard_review_mode_pauses_at_first_gate(tmp_path):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "idea": "a local dashboard review run",
            "profile": "__custom__",
            "llm_option": "fake",
            "image_option": "fake",
            "video_option": "fake",
            "vlm_option": "fake",
            "style": "anime",
            "format": "short_film",
            "language": "en",
            "run_mode": "review",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/runs",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert status["jobs"][0]["status"] == "paused"
    assert status["jobs"][0]["auto"] is False
    # The style stage runs first, so review mode now pauses at the style gate.
    assert status["jobs"][0]["current_stage"] == "style"


def test_http_dashboard_run_accepts_initial_reference_uploads(tmp_path):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    boundary = "----studio-agent-new-run-reference-test"

    def field(name, value):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    def file_part(name, filename, data):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + data + b"\r\n"

    body = b"".join([
        field("idea", "a local dashboard reference intake run"),
        field("profile", "__custom__"),
        field("llm_option", "fake"),
        field("image_option", "fake"),
        field("video_option", "fake"),
        field("vlm_option", "fake"),
        field("style", "anime"),
        field("format", "short_film"),
        field("language", "en"),
        field("run_mode", "review"),
        field("reference_note", "first image is protagonist; second image is the apartment"),
        file_part("reference_images", "hero.png", PNG_BYTES),
        file_part("reference_images", "apartment.png", PNG_BYTES),
        f"--{boundary}--\r\n".encode("utf-8"),
    ])
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/runs",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert "Started custom-model-mix" in page
    project = Project.load(tmp_path / status["jobs"][0]["project_id"])
    manifest = json.loads(project.path("references", "references.json").read_text())
    records = manifest["references"]
    assert [record["alias"] for record in records] == ["@image1", "@image2"]
    assert records[0]["pending_target"] == {"kind": "character_role", "value": "protagonist"}
    assert records[1]["pending_target"] == {"kind": "location", "value": "apartment"}


def test_http_project_state_json_reports_version(tmp_path):
    p = Project.create("http state json", root=tmp_path, stages=["plot"])
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        state = json.loads(
            urllib.request.urlopen(  # noqa: S310
                f"http://127.0.0.1:{port}/projects/{p.project_id}/state.json",
                timeout=30,
            ).read().decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert state["id"] == p.project_id
    assert state["current_stage"] == "plot"
    assert int(state["version"]) > 0


def test_http_delete_project_removes_only_selected_project(tmp_path):
    selected = Project.create("delete selected", root=tmp_path, stages=["plot"])
    neighbor = Project.create("keep neighbor", root=tmp_path, stages=["plot"])
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{selected.project_id}/delete",
            data=b"",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            response = opener.open(req, timeout=30)  # noqa: S310
            code = response.getcode()
            location = response.headers.get("Location")
        except urllib.error.HTTPError as exc:
            code = exc.code
            location = exc.headers.get("Location")
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert code == 303
    assert location == f"/?notice=Deleted%20project%3A%20{selected.project_id}"
    assert not selected.dir.exists()
    assert neighbor.dir.is_dir()


@pytest.mark.parametrize("job_status", ["queued", "running"])
def test_http_delete_project_refuses_active_job(tmp_path, job_status):
    project = Project.create("active delete protection", root=tmp_path, stages=["plot"])
    jobs = JobRunner(inline=True)
    job = RunJob(
        id="active-job",
        idea=project.idea,
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
        status=job_status,
        project_id=project.project_id,
    )
    jobs._jobs[job.id] = job
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/delete",
            data=b"",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert project.dir.is_dir()
    assert "currently generating" in body


def test_delete_project_rejects_traversal(tmp_path):
    outside = tmp_path.parent / "outside-project"
    outside.mkdir()
    outside.joinpath("project.json").write_text("{}")

    with pytest.raises(ValueError, match="project id"):
        delete_project(tmp_path, "../outside-project")

    assert outside.is_dir()


def test_http_resume_runs_in_background_and_redirects(tmp_path):
    from studio_agent import cli

    jobs = JobRunner(inline=True)
    start = jobs.start_run(
        tmp_path,
        idea="a resume-over-http test",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=False,
    )
    project = Project.load(tmp_path / start.project_id)
    # Approve the leading style gate so the resume runs the next stage (plot).
    cli._machine().approve(project)
    assert project.current_stage == "plot"

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/resume",
            data=b"",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            resp = opener.open(req, timeout=30)  # noqa: S310
            code = resp.getcode()
            location = resp.headers.get("Location")
        except urllib.error.HTTPError as exc:
            code = exc.code
            location = exc.headers.get("Location")
        status = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/jobs.json", timeout=30)  # noqa: S310
            .read()
            .decode("utf-8")
        )
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert code == 303
    assert location == f"/projects/{project.project_id}"
    resume_jobs = [j for j in status["jobs"] if j["kind"] == "resume"]
    assert resume_jobs and resume_jobs[0]["project_id"] == project.project_id
    advanced = Project.load(tmp_path / project.project_id)
    assert advanced.stage_status("plot") == "complete"


def test_project_artifacts_include_text_and_media(tmp_path):
    p = Project.create("artifact test", root=tmp_path, stages=["plot"])
    p.path("story", "plot.json").write_text(json.dumps({"ok": True}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"\0\0\0\0ftyp")

    artifacts = project_artifacts(p)
    by_path = {a["path"]: a["kind"] for a in artifacts}

    assert by_path["story/plot.json"] == "text"
    assert by_path[f"output/{p.project_id}.mp4"] == "media"


def test_media_response_forces_browser_revalidation(tmp_path):
    """Regenerated bible/keyframe images live at a stable media URL, so the
    response must tell the browser to revalidate or the old image is shown."""
    project = Project.create("media cache", root=tmp_path, stages=["bible"])
    ref = project.path("bible", "characters", "gull", "reference.png")
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_bytes(b"\x89PNG\r\n\x1a\n original")
    jobs = JobRunner(inline=False)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        url = (
            f"http://127.0.0.1:{port}/projects/{project.project_id}"
            "/media?path=bible/characters/gull/reference.png"
        )
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            cache_control = (resp.headers.get("Cache-Control") or "").lower()
            etag = resp.headers.get("ETag")
            assert resp.read() == b"\x89PNG\r\n\x1a\n original"
        assert "no-cache" in cache_control or "no-store" in cache_control
        assert etag

        # A cached client revalidates with If-None-Match and gets 304 while unchanged.
        # urllib raises HTTPError for the (bodyless) 304 response.
        cached = urllib.request.Request(url, headers={"If-None-Match": etag})
        with pytest.raises(urllib.error.HTTPError) as not_modified:
            urllib.request.urlopen(cached, timeout=30)  # noqa: S310
        assert not_modified.value.code == 304

        # After regeneration overwrites the file in place, the stale ETag no longer
        # matches, so the browser is served the new image instead of a 304.
        ref.write_bytes(b"\x89PNG\r\n\x1a\n REGENERATED")
        with urllib.request.urlopen(cached, timeout=30) as resp:  # noqa: S310
            assert resp.status == 200
            assert resp.read() == b"\x89PNG\r\n\x1a\n REGENERATED"
    finally:
        server.shutdown()
        thread.join(timeout=30)


def test_project_state_exposes_reload_version(tmp_path):
    p = Project.create("state version", root=tmp_path, stages=["plot"])

    state = project_state(p)

    assert state["id"] == p.project_id
    assert state["status"] == "in_progress"
    assert state["current_stage"] == "plot"
    assert int(state["version"]) > 0


def test_safe_project_path_blocks_traversal(tmp_path):
    p = Project.create("safe path", root=tmp_path, stages=["plot"])

    with pytest.raises(ValueError):
        safe_project_path(p, "../outside.txt")


def test_render_project_exposes_controls_without_automated_qc_summary(tmp_path):
    p = _full_pipeline_project(tmp_path)

    plot_html = render_project(p, stage="plot")
    assert "STUDIO_PROJECT_STATE_URL" in plot_html
    assert "Stage Workbench" in plot_html
    assert "A memory thief learns restraint." in plot_html
    assert "Ask for changes" in plot_html and plot_html.count("/adjust-artifact") == 1
    # global controls live once each in the console
    for endpoint in ("/resume", "/resume-auto"):
        assert endpoint in plot_html
    # approve only in the tool rail, never the console
    assert "Approve this stage" in plot_html
    assert "Approve Gate" not in plot_html
    # removed duplicated surfaces
    assert "Artifact Studio" not in plot_html
    assert "class='page-nav'" not in plot_html and "studio-nav" not in plot_html
    # polling boot survives in the page
    assert "studioStartProjectPolling" in plot_html

    # artifacts reachable by selecting their stage
    assert "INT. SCHOOL - DAY" in render_project(p, stage="script")
    bible_html = render_project(p, stage="bible")
    assert "School Hallway" in bible_html and "Upload reference" in bible_html
    assemble_html = render_project(p, stage="assemble")
    assert "all_passed" not in assemble_html
    assert "QC Summary" not in assemble_html


def test_render_project_shows_reference_alias_status_and_retarget_form(tmp_path):
    p = Project.create("reference review surface", root=tmp_path, stages=["plot", "bible"])
    save_reference_intake_uploads(
        p,
        [{"filename": "unknown.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
    )

    html = render_project(p, stage="bible")

    assert "@image1" in html
    assert "unresolved" in html
    assert "retarget-reference" in html
    assert "Attach reference" in html
    assert "Auto-detect each image" in html
    assert html.count("Auto-detect each image") == 1
    assert "name='reference_images'" in html
    assert "multiple" in html
    assert "data-reference-picker" in html
    assert "data-reference-note='note'" in html
    assert "data-alias-start='2'" in html


def test_reference_view_forms_carry_selected_stage(tmp_path):
    # The references uploader/retarget forms must round-trip the stage the user is
    # viewing, so the post-upload re-render stays on that stage instead of snapping to
    # current_stage (where the bible-gated references section would not render).
    from studio_agent.web import _render_references_view

    p = Project.create("ref stage carry", root=tmp_path, stages=["plot", "bible", "clip"])
    save_reference_intake_uploads(
        p,
        [{"filename": "hero.png", "data": PNG_BYTES, "content_type": "image/png"}],
        note="",
    )

    html = _render_references_view(p, stage="bible")

    assert "name='stage'" in html
    assert "value='bible'" in html
    # Both the upload form and the per-reference retarget form carry the stage.
    assert html.count("name='stage'") >= 2


def test_http_reference_upload_keeps_user_on_the_bible_view(tmp_path):
    # Regression: uploading a reference from the bible view after the gate has advanced must
    # re-render the bible view (where the references section lives), not snap back to
    # current_stage. Before the fix the POST ignored the posted stage and the just-uploaded
    # image vanished off the page.
    p = _full_pipeline_project(tmp_path)
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.save()

    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    boundary = "----studio-agent-stage"
    body = b"".join([
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="target"\r\n\r\n'
            "auto\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="stage"\r\n\r\n'
            "bible\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="reference_images"; filename="late.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8"),
        PNG_BYTES,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ])
    port = server.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/upload-reference",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert "Uploaded 1 reference" in page
    # The returned page is the bible workbench (selected frame == bible), so the references
    # section — and the newly uploaded image — render.
    match = re.search(r"frame[^']*selected' href='[^']*\?stage=([^']+)'", page)
    assert match and match.group(1) == "bible"
    assert "upload-reference" in page


def test_project_page_has_no_duplicate_sections(tmp_path):
    from studio_agent.web import render_project
    p = _full_pipeline_project(tmp_path)
    html = render_project(p, stage="bible")
    # Duplicated surfaces are gone.
    assert "Artifact Studio" not in html
    assert "class='page-nav'" not in html and "studio-nav" not in html
    assert "id='overview'" not in html and "id='shots'" not in html
    # Each control appears exactly once.
    assert html.count("/approve") == 1
    assert html.count("Adjustment history") == 1   # _render_adjustment_history heading
    # On the bible stage the shared "Ask for changes" sidebar box is suppressed;
    # per-card inline controls take its place (see test_bible_workbench_renders_inline_controls_and_banner).
    assert html.count("/adjust-artifact") == 0
    # Raw files still reachable, now collapsed.
    assert "<details" in html and "All files" in html
    # Preserved global controls remain.
    for endpoint in ("/resume", "/resume-auto"):
        assert endpoint in html


def test_render_project_shows_running_stage_loader(tmp_path):
    p = Project.create("running loader", root=tmp_path, stages=["plot", "script"])
    p.set_stage_status("plot", "running")

    html = render_project(p)

    assert "stage-workbench is-loading" in html
    assert "Generating current stage..." in html
    assert "This page will refresh when new files are ready." in html
    assert "stage-spinner" in html
    assert "stage-progress" in html


def test_render_project_clip_stage_shows_generated_keyframes(tmp_path):
    p = Project.create(
        "clip stage workbench",
        root=tmp_path,
        stages=["concept", "bible", "clip", "keyframes", "video", "review", "assemble"],
    )
    p.current_stage = "keyframes"
    p.set_stage_status("keyframes", "complete")
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "description": "A dancer crosses a neon hallway.",
        "camera": "wide",
        "action": "The dancer spins toward camera.",
        "duration_s": 15,
        "keyframe": "sh-001.png",
    }]}))
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(PNG_BYTES)
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text("keyframe prompt")

    html = render_project(p, stage="keyframes")

    assert "A dancer crosses a neon hallway." in html
    assert "media?path=storyboard%2Fkeyframes%2Fsh-001.png" in html
    assert "Project complete." not in html


def test_native_audio_workbench_renders_prompt_track_source_and_manual_review_note(tmp_path):
    p = Project.create(
        "native audio workbench",
        root=tmp_path,
        stages=["audio", "assemble"],
        model_config={"audio_mode": "native_video"},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001", "keyframe": "sh-001.png", "duration_s": 2,
    }]}))
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("## Sound")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip")
    p.path("assets", "audio", "sh-001.native.wav").write_bytes(b"wav")
    p.path("assets", "qc", "sh-001.json").write_text("{}")

    html = render_project(p)

    for rel in (
        "storyboard/prompts/sh-001.video.md",
        "assets/audio/sh-001.native.wav",
        "assets/clips/sh-001.mp4",
    ):
        assert rel in html
    assert "Extracted tracks" in html
    assert "Source clips" in html
    assert "Review dialogue, effects, ambience, and music absence manually." in html
    assert "QC reports" not in html


def test_native_audio_workbench_escapes_and_url_encodes_artifact_links(tmp_path):
    p = Project.create(
        "safe native audio links",
        root=tmp_path,
        stages=["audio"],
        model_config={"audio_mode": "native_video"},
    )
    p.path("assets", "audio", "unsafe & track.native.wav").write_bytes(b"wav")

    html = render_project(p)

    assert "unsafe &amp; track.native.wav" in html
    assert "unsafe%20%26%20track.native.wav" in html
    assert "unsafe & track.native.wav" not in html


def test_render_project_labels_cost_as_estimate_and_warns_for_legacy_history(tmp_path):
    p = Project.create("estimated cost ui", root=tmp_path, stages=["audio"])
    p.add_cost(stage="plot", provider="remote", cost_usd=0.0, seconds=1.0)

    html = render_project(p)

    assert "Estimated cost" in html
    assert "Historical cost entries may be incomplete" in html


def test_http_reference_upload_accepts_multiple_images_and_invalidates_downstream(tmp_path):
    p = Project.create(
        "web upload reference",
        root=tmp_path,
        stages=["plot", "storyboard", "video", "review", "audio", "assemble"],
    )
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    boundary = "----studio-agent-test"
    def file_part(filename, data):
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="reference_images"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode("utf-8") + data + b"\r\n"

    body = b"".join([
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="target"\r\n\r\n'
        "auto\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="label"\r\n\r\n'
        "mixed references\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="note"\r\n\r\n'
        "analyze each image\r\n".encode("utf-8"),
        file_part("mara-face.png", PNG_BYTES),
        file_part("apartment-location.png", PNG_BYTES),
        f"--{boundary}--\r\n".encode("utf-8"),
    ])
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/upload-reference",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    manifest = json.loads(updated.path("references", "references.json").read_text())
    records = manifest["references"]
    assert "Uploaded 2 references" in page
    assert [record["alias"] for record in records] == ["@image1", "@image2"]
    assert [(record["target_type"], record["target_id"]) for record in records] == [
        ("character", "Mara"),
        ("location", "Apartment"),
    ]
    assert all(record["label"] == "mixed references" for record in records)
    assert all(updated.path(*record["path"].split("/")).read_bytes() == PNG_BYTES for record in records)
    assert updated.current_stage == "storyboard"
    assert updated.stage_status("plot") == "approved"
    assert updated.stage_status("storyboard") == "pending"


def test_single_revise_form_posts_to_adjust_artifact(tmp_path):
    from studio_agent.web import _render_workbench_revision
    from studio_agent.stage_workbench import workbench_context
    p = _full_pipeline_project(tmp_path)
    form = _render_workbench_revision(p, workbench_context(p, stage="plot"))
    assert "Revise current stage" in form        # the workbench revise form actually rendered
    assert "/adjust-artifact" in form            # it posts to the unified endpoint
    assert "/workbench-revise" not in form        # old endpoint retired from the form


def test_http_workbench_revise_updates_current_stage_artifact(tmp_path):
    p = Project.create(
        "web workbench revise",
        root=tmp_path,
        stages=["plot", "script"],
        model_config={"llm": "fake"},
    )
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "old",
        "synopsis": "old synopsis",
        "themes": [],
    }))
    p.set_stage_status("plot", "complete")
    p.current_stage = "plot"
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "path": "story/plot.json",
            "instruction": "make it kinder",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/adjust-artifact",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    plot = json.loads(updated.path("story", "plot.json").read_text())
    assert "Revised story/plot.json" in page
    assert plot["revision_note"] == "make it kinder"
    assert updated.current_stage == "plot"
    assert updated.stage_status("script") == "pending"


def test_http_workbench_revise_empty_instruction_shows_notice(tmp_path):
    p = Project.create(
        "web workbench empty revise",
        root=tmp_path,
        stages=["plot", "script"],
        model_config={"llm": "fake"},
    )
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "old",
        "synopsis": "old synopsis",
        "themes": [],
    }))
    p.set_stage_status("plot", "complete")
    p.current_stage = "plot"
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "path": "story/plot.json",
            "instruction": "",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/adjust-artifact",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    assert "Tell the LLM what to change" in page
    assert "revision_note" not in updated.path("story", "plot.json").read_text()
    assert updated.current_stage == "plot"


def test_http_raw_artifact_save_invalidates_downstream(tmp_path):
    """Hand-editing via the raw 'Save Artifact' editor must cascade like Revise does."""
    p = Project.create(
        "web raw artifact save",
        root=tmp_path,
        stages=["plot", "script"],
        model_config={"llm": "fake"},
    )
    p.path("story", "plot.json").write_text(json.dumps({
        "logline": "old",
        "synopsis": "old synopsis",
        "themes": [],
    }))
    # A downstream derivative exists; editing the plot must archive it, not destroy it.
    p.path("story", "script.json").write_text(json.dumps({"scenes": ["derived from old plot"]}))
    # Both stages already past their gates — we want to watch the later one go stale.
    p.set_stage_status("plot", "approved")
    p.set_stage_status("script", "approved")
    p.current_stage = None
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    new_plot = json.dumps({"logline": "new", "synopsis": "new synopsis", "themes": []})
    try:
        data = urllib.parse.urlencode({
            "path": "story/plot.json",
            "content": new_plot,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/artifact",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30).read()  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # The new text replaced the old (and the old is archived under history/, not destroyed).
    assert json.loads(updated.path("story", "plot.json").read_text())["logline"] == "new"
    # The stale downstream derivative was archived under history/ (reversible), not left in place.
    assert not updated.path("story", "script.json").is_file()
    archived = list(updated.dir.joinpath("history").rglob("script.json"))
    assert archived, "stale downstream script.json should be archived under history/"
    # Downstream stage is now stale and re-runnable; the edited stage is back at its gate.
    assert updated.stage_status("script") == "pending"
    assert updated.stage_status("plot") == "complete"
    assert updated.current_stage == "plot"


def test_cutting_room_styles_present(tmp_path):
    from studio_agent.web import render_project, CSS
    p = _project_with_shot(tmp_path, qc={"overall_pass": True, "summary": "ok"})
    render_project(p)  # smoke: still renders
    assert "--amber:#F2A33C" in CSS
    assert ".frame.selected" in CSS
    assert "prefers-reduced-motion" in CSS
    assert "--warn:" in CSS
    assert "--info:" in CSS
    assert "busy-progress" in CSS


def test_http_workbench_regenerate_asset_archives_selected_shot_and_self_starts(tmp_path):
    """shot_video regenerate archives only sh-001's clip, then self-starts the video stage."""
    p = Project.create(
        "web workbench regen",
        root=tmp_path,
        stages=["video", "review", "audio", "assemble"],
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "duration_s": 3,
         "keyframe_prompt": "test shot one", "reference_seed": 1},
        {"id": "sh-002", "keyframe": "sh-002.png", "duration_s": 3,
         "keyframe_prompt": "test shot two", "reference_seed": 2},
    ]}))
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "video"
    p.status = "in_progress"
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("storyboard", "keyframes", "sh-002.png").write_bytes(b"\x89PNG\r\n")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip one")
    p.path("assets", "clips", "sh-002.mp4").write_bytes(b"clip two")
    # shot_video regeneration re-confirms the compiled video prompt, and the self-started video
    # stage runs every shot, so both shots need their prompt files on disk.
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("Close-up: fist tightens.")
    p.path("storyboard", "prompts", "sh-002.video.md").write_text("Wide: the room settles.")
    p.path("assets", "qc", "summary.json").write_text(json.dumps({"ok": True}))
    p.path("edit", "timeline.json").write_text(json.dumps({"ok": True}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"final")
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({
            "kind": "shot_video",
            "shot_id": "sh-001",
            "rel_path": "assets/clips/sh-001.mp4",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/regenerate-asset",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=30).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # Handler dispatches in background and 303-redirects; inline job completes before redirect.
    assert status == 303
    assert updated.path("assets", "clips", "sh-001.mp4").is_file()   # recreated
    assert updated.path("assets", "clips", "sh-002.mp4").is_file()   # untouched
    assert updated.current_stage == "video"
    assert updated.stage_status("video") == "complete"               # self-start completed


def test_http_regenerate_action_self_starts_in_background_and_pauses_at_gate(tmp_path):
    """The review-panel /regenerate action must dispatch a background run (303 redirect,
    loading sign) and pause at the video gate (auto=False), not run synchronously to the
    end of the pipeline."""
    p = Project.create(
        "web regenerate action",
        root=tmp_path,
        stages=["video", "review", "audio", "assemble"],
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "duration_s": 3,
         "keyframe_prompt": "test shot one", "reference_seed": 1},
    ]}))
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "video"
    p.status = "in_progress"
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip one")
    # The /regenerate action re-confirms the compiled video prompt, so it must exist on disk.
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("Close-up: fist tightens.")
    p.path("assets", "qc", "summary.json").write_text(json.dumps({"ok": True}))
    p.path("edit", "timeline.json").write_text(json.dumps({"ok": True}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"final")
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({"shot": "sh-001"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/regenerate",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=30).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # Dispatched in the background and 303-redirected (consistent loading sign), not a
    # synchronous 200 inline render.
    assert status == 303
    assert updated.path("assets", "clips", "sh-001.mp4").is_file()  # recreated
    assert updated.current_stage == "video"
    assert updated.stage_status("video") == "complete"  # self-start completed the stage
    # auto=False paused at the video gate; downstream stages were NOT run to completion.
    assert updated.stage_status("assemble") != "complete"


def test_http_keyframe_feedback_revises_prompt_and_marks_storyboard_pending(tmp_path):
    from studio_agent.prompt_approvals import confirm_prompt_batch

    p = Project.create(
        "web keyframe feedback",
        root=tmp_path,
        stages=["storyboard", "video", "review", "audio", "assemble"],
        # Full fake providers: the self-started keyframes stage regenerates the image, so the
        # image/video providers must be fake too (not just the LLM).
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "keyframe_prompt": "old prompt one"},
        {"id": "sh-002", "keyframe": "sh-002.png", "keyframe_prompt": "old prompt two"},
    ]}))
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "storyboard"
    p.status = "in_progress"
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text("old prompt one")
    p.path("storyboard", "prompts", "sh-002.keyframe.md").write_text("old prompt two")
    # Resolve the creative-direction decision the keyframes preflight needs.
    p.path("story", "creative_brief.json").write_text(json.dumps({
        "visual_direction": {"camera_language": "patient and intimate"},
        "hard_avoidances": [],
    }))
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"kf one")
    p.path("storyboard", "keyframes", "sh-002.png").write_bytes(b"kf two")
    # A completed keyframes gate has the whole prompt batch confirmed; the self-started
    # regeneration re-confirms only the revised shot within that batch.
    confirm_prompt_batch(p, "keyframes", confirmer="human")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip one")
    p.path("assets", "clips", "sh-002.mp4").write_bytes(b"clip two")
    p.path("assets", "qc", "summary.json").write_text(json.dumps({"ok": True}))
    p.path("edit", "timeline.json").write_text(json.dumps({"ok": True}))
    p.path("output", f"{p.project_id}.mp4").write_bytes(b"final")
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({
            "shot_id": "sh-001",
            "feedback": "make it less photoreal and keep the red coat",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/keyframe-feedback",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=30).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    prompt = updated.path("storyboard", "prompts", "sh-001.keyframe.md").read_text()
    shots = json.loads(updated.path("storyboard", "shots.json").read_text())["shots"]
    shot_one = next(shot for shot in shots if shot["id"] == "sh-001")
    # Handler dispatches in background and 303-redirects; inline job completes before redirect.
    assert status == 303
    assert "less photoreal" in prompt
    assert shot_one["keyframe_prompt"] == prompt.strip()
    # Keyframe feedback invalidates the now-separate keyframes stage and the self-started run
    # regenerates sh-001's image, leaving sh-002 untouched and pausing at the keyframes gate.
    assert updated.path("storyboard", "keyframes", "sh-001.png").is_file()
    assert updated.path("storyboard", "keyframes", "sh-002.png").is_file()
    assert updated.current_stage == "keyframes"
    assert updated.stage_status("keyframes") == "complete"


def test_http_video_feedback_revises_motion_prompt_without_rerender(tmp_path):
    p = Project.create(
        "web video feedback",
        root=tmp_path,
        stages=["storyboard", "video", "review", "audio", "assemble"],
        model_config={"llm": "fake"},
    )
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png"},
    ]}))
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "video"
    p.status = "in_progress"
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("old motion one")
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"kf one")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"clip one")
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({
            "shot_id": "sh-001",
            "feedback": "slow the camera push and keep the dialogue exact",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/video-feedback",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=30).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    prompt = updated.path("storyboard", "prompts", "sh-001.video.md").read_text()
    assert status in (200, 303)
    assert "slow the camera push" in prompt
    # Revise only: the paid clip is untouched until the user clicks Regenerate (invariant #7).
    assert updated.path("assets", "clips", "sh-001.mp4").is_file()
    assert updated.stage_status("video") == "approved"


def test_render_project_shows_video_feedback_form(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)

    html = render_project(p)

    assert "/video-feedback" in html
    assert "Revise motion prompt" in html


def _add_motion(p):
    shots_path = p.path("storyboard", "shots.json")
    data = json.loads(shots_path.read_text())
    data["shots"][0]["camera"] = "Sony Venice 50mm"
    data["shots"][0]["camera_movement"] = "缓慢dolly推近（从2m推至1.2m）"
    shots_path.write_text(json.dumps(data, ensure_ascii=False))
    p.path("storyboard", "prompts", "sh-001.video.md").write_text("缓慢dolly推近 ...")


def test_shot_reviews_carries_camera_movement_and_motion_prompt(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)
    _add_motion(p)

    review = shot_reviews(p)[0]

    assert review["camera"] == "Sony Venice 50mm"
    assert review["camera_movement"] == "缓慢dolly推近（从2m推至1.2m）"
    assert review["video_prompt"] == "storyboard/prompts/sh-001.video.md"


def test_review_card_shows_camera_movement_and_motion_prompt_link(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)
    _add_motion(p)

    html = render_project(p)

    # The 运镜 is visible on the review card, with a link to the full motion prompt.
    assert "缓慢dolly推近（从2m推至1.2m）" in html
    assert "sh-001.video.md" in html


def test_shot_reviews_video_prompt_absent_when_not_generated(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)  # no <id>.video.md on disk

    review = shot_reviews(p)[0]

    assert review["video_prompt"] is None


def test_render_project_shows_storyboard_keyframe_feedback_form(tmp_path):
    p = Project.create(
        "web keyframe feedback form",
        root=tmp_path,
        stages=["storyboard", "keyframes"],
        model_config={"llm": "fake"},
    )
    p.current_stage = "keyframes"
    p.set_stage_status("keyframes", "complete")
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "duration_s": 3,
        "keyframe": "sh-001.png",
        "action": "Mara turns away.",
    }]}))
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text("old prompt")
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"kf")
    p.save()

    html = render_project(p)

    assert "/keyframe-feedback" in html
    assert "name='feedback'" in html
    assert "Revise prompt &amp; regenerate keyframe" in html
    assert "keyframe prompt" in html


def test_shot_reviews_uses_manual_review_even_when_legacy_qc_exists(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)

    reviews = shot_reviews(p)

    assert len(reviews) == 1
    r = reviews[0]
    assert r["id"] == "sh-001"
    assert r["clip"] == "assets/clips/sh-001.mp4"
    assert r["keyframe"] == "storyboard/keyframes/sh-001.png"
    assert r["review_mode"] == "manual"
    assert "verdict" not in r
    assert "failures" not in r


def test_render_project_shows_per_shot_clip_and_regenerate_button(tmp_path):
    p = _project_with_shot(tmp_path, qc=_FAILING_QC)

    html = render_project(p)

    assert "sh-001" in html
    # The clip is shown inline for a human decision; legacy QC is not rendered.
    assert "assets/clips/sh-001.mp4" in html
    assert "MANUAL REVIEW" in html
    assert "FAIL" not in html
    assert "identity_drift" not in html
    # A one-click per-shot regenerate button carrying the shot id.
    assert "value='sh-001'" in html or 'value="sh-001"' in html


def test_render_project_ignores_legacy_qc_warning_state(tmp_path):
    p = _project_with_shot(tmp_path, qc={
        "overall_pass": True,
        "max_severity": "low",
        "recommendation": "approve_with_warnings",
        "summary": "minor texture shimmer",
        "checks": [
            {"dimension": "artifacts", "passed": False, "severity": "low",
             "detail": "minor texture shimmer", "timestamp": "00:01"},
        ],
    })

    html = render_project(p)

    assert "MANUAL REVIEW" in html
    assert "WARN" not in html
    assert "minor texture shimmer" not in html
    assert "FAIL" not in html


def test_shot_reviews_handles_missing_qc(tmp_path):
    p = Project.create("noqc", root=tmp_path, stages=["video"])
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [
        {"id": "sh-001", "keyframe": "sh-001.png", "duration_s": 2.0}
    ]}))

    reviews = shot_reviews(p)

    assert reviews[0]["review_mode"] == "manual"
    assert reviews[0]["clip"] is None


def test_characters_view_shows_full_identity_board(tmp_path):
    """The bible card surfaces each distinct identity-board field, not just one line."""
    p = Project.create("ident board", root=tmp_path, stages=["bible"])
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({
        "name": "Mara",
        "seed": 7,
        "description": "A weathered lighthouse keeper.",
        "wardrobe": "oilskin coat",
        "palette": "stormy blues",
    }))
    (cdir / "identity_board.json").write_text(json.dumps({
        "canonical_face": "angular face, deep-set grey eyes",
        "canonical_body": "tall, broad-shouldered build",
        "hair": "close-cropped silver hair",
        "wardrobe": "oilskin coat",
        "palette": "stormy blues",
        "do": ["keep the scar over the left brow"],
        "dont": ["change the coat colour"],
        "prompt_aliases": ["Mara", "the keeper"],
    }))

    html = render_project(p)

    assert "angular face, deep-set grey eyes" in html
    assert "tall, broad-shouldered build" in html
    assert "close-cropped silver hair" in html
    assert "the keeper" in html
    assert "keep the scar over the left brow" in html
    assert "change the coat colour" in html


def test_character_card_renders_dict_description_as_points_not_raw_repr(tmp_path):
    """A nested-dict character description must render as readable point form, never as a
    raw Python dict repr blob."""
    p = Project.create("dict desc", root=tmp_path, stages=["bible"])
    cdir = p.path("bible", "characters", "lin")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({
        "name": "林",
        "seed": 1,
        "description": {
            "canonical_face": "young asian male, long face",
            "canonical_body": "slim build, about 1.8m",
            "hair_and_grooming": "short blond dyed hair",
            "wardrobe_lock": "black tee, black jeans",
            "palette": {"colors": ["#1A1A1A"], "description": "low saturation"},
        },
    }, ensure_ascii=False))

    html = render_project(p)

    # No raw python dict repr leaking into the page. The raw repr keeps the underscore
    # keys; the humanized form spaces and capitalizes them ("Canonical face").
    assert "canonical_face" not in html
    assert "wardrobe_lock" not in html
    assert "hair_and_grooming" not in html
    # Humanized values appear in readable form.
    assert "young asian male, long face" in html
    assert "short blond dyed hair" in html
    assert "black tee, black jeans" in html


def test_location_card_renders_palette_and_points_not_raw_repr(tmp_path):
    """A location with list-valued palette/materials/props must render as point form and
    color chips, never as raw Python list reprs."""
    p = Project.create("loc fmt", root=tmp_path, stages=["bible"])
    ldir = p.path("bible", "locations", "field")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": "dry field",
        "description": "a flat expanse of cracked earth",
        "palette": ["#C2A67A", "#A67B5B"],
        "lighting": "harsh midday sun",
        "materials": ["dry soil", "cracked mud"],
        "hero_props": ["Y-shaped crack center"],
        "continuity_rules": ["crack pattern identical across shots"],
    }, ensure_ascii=False))

    html = render_project(p)

    # No python list repr leaking into the page.
    assert "['#C2A67A'" not in html
    assert "['dry soil'" not in html
    # Palette colors and point-form details are surfaced.
    assert "#C2A67A" in html
    assert "dry soil" in html
    assert "Y-shaped crack center" in html
    assert "crack pattern identical across shots" in html


def test_bible_workbench_labels_character_image_roles(tmp_path):
    # Each character now has ONE combined model sheet (reference.png).
    # Turnaround and Expressions labels are removed from the UI.
    p = Project.create("caption workbench", root=tmp_path, stages=["bible"])
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({"name": "Mara"}))
    (cdir / "identity_board.json").write_text("{}")
    (cdir / "reference.png").write_bytes(PNG_BYTES)
    (cdir / "states.json").write_text(json.dumps({"states": [{
        "id": "werewolf",
        "label": "Werewolf",
        "kind": "endpoint",
        "appearance_changes": ["species", "anatomy"],
        "reference_required": True,
        "reference_image": "states/werewolf/reference.png",
    }]}))
    state_dir = cdir / "states" / "werewolf"
    state_dir.mkdir(parents=True)
    (state_dir / "reference.png").write_bytes(PNG_BYTES)
    p.set_stage_status("bible", "complete")
    p.save()

    html = render_project(p)

    assert "<figcaption>Identity reference</figcaption>" in html
    assert "<figcaption>Appearance state reference: Werewolf</figcaption>" in html
    assert "<figcaption>Turnaround</figcaption>" not in html
    assert "<figcaption>Expressions</figcaption>" not in html


def test_character_overview_labels_character_image_roles(tmp_path):
    # Each character card shows one combined sheet with a "Reference" label.
    # Turnaround and Expressions figcaptions are no longer emitted.
    p = Project.create("caption overview", root=tmp_path, stages=["bible"])
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({"name": "Mara"}))
    (cdir / "reference.png").write_bytes(PNG_BYTES)

    html = render_project(p)

    assert html.count("<figcaption>Identity reference</figcaption>") >= 1
    assert html.count("<figcaption>Turnaround</figcaption>") == 0
    assert html.count("<figcaption>Expressions</figcaption>") == 0


def test_character_card_shows_single_combined_sheet(tmp_path):
    """Character card shows one combined sheet label; no turnaround/expressions figcaptions.

    Even if legacy turnaround.png / expressions.png files exist on disk, the UI
    must only show the Reference label — Turnaround and Expressions are gone.
    """
    p = Project.create("combined sheet char", root=tmp_path, stages=["bible"])
    cdir = p.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({"name": "Mara"}))
    (cdir / "identity_board.json").write_text("{}")
    # Write all three legacy files to force the old code path to surface them.
    for name in ("reference.png", "turnaround.png", "expressions.png"):
        (cdir / name).write_bytes(PNG_BYTES)
    p.set_stage_status("bible", "complete")
    p.save()

    html = render_project(p)

    assert "<figcaption>Identity reference</figcaption>" in html
    assert "<figcaption>Turnaround</figcaption>" not in html
    assert "<figcaption>Expressions</figcaption>" not in html


def test_location_card_shows_single_combined_sheet(tmp_path):
    """Location card shows one combined sheet label; no environment_board figcaption.

    Even if a legacy environment_board.png file exists on disk, the UI must not
    show an Environment Board figcaption — only Reference is shown.
    """
    p = Project.create("combined sheet loc", root=tmp_path, stages=["bible"])
    ldir = p.path("bible", "locations", "lighthouse")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({"name": "Lighthouse"}))
    # Write both legacy files so the old code would have included environment_board.
    for name in ("reference.png", "environment_board.png"):
        (ldir / name).write_bytes(PNG_BYTES)

    html = render_project(p)

    assert "<figcaption>Reference</figcaption>" in html
    assert "<figcaption>Environment Board</figcaption>" not in html


def test_project_page_includes_image_lightbox(tmp_path):
    """Every media image is zoomable through a shared lightbox overlay."""
    p = Project.create("lightbox", root=tmp_path, stages=["bible"])

    html = render_project(p)

    assert "data-lightbox" in html
    assert "studioInitLightbox" in BUSY_JS


def test_http_regenerate_bible_asset_rebuilds_image_and_pauses_at_bible(tmp_path):
    jobs = JobRunner(inline=True)
    job = jobs.start_run(
        tmp_path,
        idea="bible regen smoke",
        profile_name="fake",
        style_name="anime",
        format_name="short_film",
        language="en",
        auto=True,
    )
    assert job.status == "done"
    project = Project.load(tmp_path / job.project_id)
    chars = project.path("bible", "characters")
    slug = sorted(c.name for c in chars.iterdir() if c.is_dir())[0]
    rel = f"bible/characters/{slug}/reference.png"
    assert (project.dir / rel).is_file()

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({
            "kind": "bible_asset",
            "rel_path": rel,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/regenerate-asset",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=60).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / project.project_id)
    # Handler dispatches in background and 303-redirects; inline job completes before redirect.
    assert status == 303
    assert (updated.dir / rel).is_file()
    assert updated.stage_status("bible") == "complete"
    assert updated.current_stage == "bible"


def test_workspace_folds_shots_overview_and_references(tmp_path):
    from studio_agent.web import _render_workbench_payload, _render_workbench_actions
    from studio_agent.stage_workbench import workbench_context
    p = _full_pipeline_project(tmp_path)

    bible = _render_workbench_payload(p, workbench_context(p, stage="bible"))
    assert "/upload-reference" in bible          # references uploader folded into bible
    assert "School Hallway" in bible             # bible payload still rendered

    video = _render_workbench_payload(p, workbench_context(p, stage="video"))
    assert "MANUAL REVIEW" in video
    assert "PASS" not in video

    assemble = _render_workbench_payload(p, workbench_context(p, stage="assemble"))
    assert "QC Summary" not in assemble and "all_passed" not in assemble

    # by-id "Regenerate shot" field appears in the video tool rail (not on other stages)
    video_actions = _render_workbench_actions(p, workbench_context(p, stage="video"))
    bible_actions = _render_workbench_actions(p, workbench_context(p, stage="bible"))
    assert "name='shot'" in video_actions
    assert "name='shot'" not in bible_actions


def test_non_gate_stage_disables_approval(tmp_path):
    from studio_agent.web import _render_workbench_approval
    from studio_agent.stage_workbench import workbench_context
    p = _full_pipeline_project(tmp_path)
    # current gate is the first stage (plot); select a later, non-gate stage
    ctx = workbench_context(p, stage="assemble")
    approval = _render_workbench_approval(p, ctx)
    assert "disabled" in approval     # approve is disabled when not the active gate


def test_zh_translations_cover_new_labels(tmp_path):
    from studio_agent.i18n import UI_STRINGS, t
    for label in ("All files", "Regenerate shot"):   # the genuinely new visible labels
        assert label in UI_STRINGS                   # has a Chinese catalog entry
    # New background-music form labels carry Chinese translations (invariant #10).
    assert t("Background music", "zh") == "背景音乐"
    assert t("Music mood (optional)", "zh") == "音乐情绪（可选）"


def test_dashboard_form_has_a_vision_model_selector():
    from studio_agent import cli
    from studio_agent.web import _render_dashboard, JobRunner
    from pathlib import Path

    html = _render_dashboard(Path("."), cli.load_config(), JobRunner(inline=True))

    assert "name='vlm_option'" in html
    assert "Vision model" in html
    # The reframed catalog no longer says "QC" in the visible labels.
    assert "QC<" not in html and "QC " not in html


def test_dashboard_remembers_model_selections_in_browser():
    from pathlib import Path

    from studio_agent import cli
    from studio_agent.web import _render_dashboard, JobRunner

    html = _render_dashboard(Path("."), cli.load_config(), JobRunner(inline=True))

    for kind in ("llm", "image", "video", "vlm"):
        assert f"data-model-preference='{kind}'" in html
    assert "studioModelPreferences:v1" in BUSY_JS
    assert "function studioRestoreModelPreferences()" in BUSY_JS
    assert "function studioSaveModelPreferences()" in BUSY_JS
    assert "studioRestoreModelPreferences();" in BUSY_JS
    assert "select.options" in BUSY_JS


def test_render_index_translates_chrome_to_chinese(tmp_path):
    from studio_agent.web import render_index
    html = render_index(tmp_path, lang="zh")
    assert "新建运行" in html        # "New Run"
    assert "任务监控" in html        # "Job Monitor"
    assert "创意" in html            # "Idea"
    # markup-anchored: the visible heading text is translated (English source keys
    # still appear inside the server-injected STUDIO_I18N map, which is fine)
    assert ">New Run<" not in html
    assert ">Job Monitor<" not in html


def test_workbench_content_panels_translate(tmp_path):
    from studio_agent.web import render_project, render_artifact
    p = _full_pipeline_project(tmp_path)
    # script panel chrome is translated; English chrome does not survive as markup
    script = render_project(p, stage="script", lang="zh")
    assert "节拍" in script and ">Beats<" not in script        # "Beats"
    assert "对白" in script and ">Dialogue<" not in script     # "Dialogue"
    # bible panel
    bible = render_project(p, stage="bible", lang="zh")
    assert "色板" in bible and ">Palette<" not in bible        # "Palette"
    assert "参考素材" in bible and ">References<" not in bible  # "References"
    # stage-workbench header label is translated
    assert ">Show Bible<" not in bible
    # artifact viewer chrome is translated
    art, _, _ = render_artifact(p, "story/plot.json", lang="zh")
    assert "返回" in art and ">Back<" not in art               # "Back"
    # default English still renders English chrome
    en = render_project(p, stage="script", lang="en")
    assert ">Beats<" in en


def test_new_run_form_model_style_length_translate(tmp_path):
    from studio_agent.web import render_index
    html = render_index(tmp_path, lang="zh")
    assert "模型选择" in html        # "Model selection"
    assert ">Model selection<" not in html
    assert "描述你的风格" in html    # "Describe your style"
    assert "时长" in html            # "Length"
    assert "光影与质感" in html      # "Light & texture" (was escaped-key mismatch before)
    assert "视觉模型（参考与风格）" in html  # "Vision model (reference & style)"


def test_render_index_english_is_default(tmp_path):
    from studio_agent.web import render_index
    html = render_index(tmp_path)
    assert "New Run" in html
    assert "新建运行" not in html


def test_language_toggle_marks_active_language_server_side(tmp_path):
    from studio_agent.web import _language_toggle
    zh = _language_toggle("zh")
    assert "data-lang='zh'" in zh and "active" in zh


def test_render_project_translates_chrome_to_chinese(tmp_path):
    from studio_agent.web import render_project
    p = _full_pipeline_project(tmp_path)
    html = render_project(p, lang="zh")
    assert "继续" in html            # "Resume"
    assert "工作台" in html          # "Workbench"
    assert ">Resume<" not in html
    # the outer shell must also carry the language, not just the body
    assert "<html lang='zh'>" in html


def test_render_project_keeps_idea_untranslated(tmp_path):
    from studio_agent.web import render_project
    p = _full_pipeline_project(tmp_path)
    html = render_project(p, lang="zh")
    assert p.idea in html            # dynamic content is not translated


def test_new_run_form_has_no_language_select(tmp_path):
    from studio_agent.web import render_index
    html = render_index(tmp_path)
    # the standalone content-language dropdown is gone; the toggle governs it
    assert "name='language'" not in html


def test_client_translator_is_removed(tmp_path):
    from studio_agent.web import BUSY_JS
    # the old client-side DOM-walking translator and its hand-kept dict are gone
    assert "STUDIO_TEXT_ZH" not in BUSY_JS
    assert "studioApplyLanguage" not in BUSY_JS
    assert "studioTranslateText" not in BUSY_JS


def test_toggle_handler_sets_cookie_and_reloads(tmp_path):
    from studio_agent.web import BUSY_JS
    assert "studioLang=" in BUSY_JS           # writes the cookie the server reads
    assert "window.location.reload" in BUSY_JS  # round-trips so the server re-renders


def test_page_injects_server_built_language_map(tmp_path):
    from studio_agent.web import render_index
    # zh ships the catalog (generated from the one Python source) for runtime-injected text
    zh = render_index(tmp_path, lang="zh")
    assert "window.STUDIO_I18N=" in zh
    assert '"lang": "zh"' in zh
    assert "重生成镜头" in zh                  # a catalog value rides along in the map
    # en ships an empty string map (nothing to translate)
    en = render_index(tmp_path, lang="en")
    assert '"lang": "en"' in en
    assert '"strings": {}' in en


# ---------------------------------------------------------------------------
# Task 4: regenerate actions self-start a run (no redundant "Resume" click)
# ---------------------------------------------------------------------------

def _storyboard_project(tmp_path):
    """Minimal project wired to run the storyboard stage with fake providers."""
    p = Project.create(
        "storyboard self-start",
        root=tmp_path,
        stages=["storyboard", "video"],
    )
    # Set all stages approved so only the invalidated stage needs to run.
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "storyboard"
    p.status = "in_progress"
    # shots.json with keyframe_prompt set to skip LLM-based prompt generation.
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "camera": "close-up",
        "action": "A student clenches a glowing fist.",
        "duration_s": 3,
        "keyframe": "sh-001.png",
        "characters": [],
        "reference_seed": 42,
        "keyframe_prompt": "close-up of a glowing fist",
    }]}))
    # Pre-write the keyframe prompt file so the storyboard stage skips the director.
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(
        "close-up of a glowing fist"
    )
    # Write the creative_brief so the storyboard preflight finds camera_language resolved.
    p.path("story", "creative_brief.json").write_text(json.dumps({
        "visual_direction": {"camera_language": "patient and intimate"},
        "hard_avoidances": [],
    }))
    # A keyframe exists so the project looks "complete" for storyboard already.
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.save()
    return p


def test_http_regenerate_shot_video_self_starts_run(tmp_path):
    """After a single shot_video regenerate POST, the clip is recreated (no Resume needed)."""
    p = Project.create(
        "shot video self start",
        root=tmp_path,
        stages=["video"],
    )
    for stage in p.stages:
        p.set_stage_status(stage, "approved")
    p.current_stage = "video"
    p.status = "in_progress"
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "camera": "close-up",
        "action": "Glowing fist.",
        "duration_s": 3,
        "keyframe": "sh-001.png",
        "characters": [],
        "reference_seed": 42,
        "keyframe_prompt": "close-up of a glowing fist",
    }]}))
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("assets", "clips", "sh-001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    # A completed pipeline has the compiled video prompt on disk; shot_video regeneration
    # re-confirms it, so the fixture must include the prompt file.
    p.path("storyboard", "prompts", "sh-001.video.md").write_text(
        "Close-up: the fist tightens as light pulses."
    )
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "kind": "shot_video",
            "shot_id": "sh-001",
            "rel_path": "assets/clips/sh-001.mp4",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/regenerate-asset",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # The self-started fake video stage recreated the clip — no second Resume POST needed.
    assert updated.path("assets", "clips", "sh-001.mp4").is_file(), (
        "clip should have been regenerated by the self-started run"
    )
    assert updated.stage_status("video") == "complete", (
        "video stage should be complete after self-started run (not pending)"
    )
    assert "Click Resume" not in page


def test_http_regenerate_keyframe_self_starts_run(tmp_path):
    """After a single keyframe regenerate POST, the keyframe is recreated (no Resume needed)."""
    p = _storyboard_project(tmp_path)
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "kind": "keyframe",
            "shot_id": "sh-001",
            "rel_path": "storyboard/keyframes/sh-001.png",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/regenerate-asset",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # The self-started fake storyboard stage recreated the keyframe.
    assert updated.path("storyboard", "keyframes", "sh-001.png").is_file(), (
        "keyframe should have been regenerated by the self-started run"
    )
    assert updated.stage_status("keyframes") == "complete", (
        "keyframes stage should be complete after self-started run (not pending)"
    )
    assert "Click Resume" not in page


def test_http_keyframe_feedback_self_starts_run(tmp_path):
    """After keyframe-feedback POST, the keyframe is regenerated without a second Resume."""
    p = _storyboard_project(tmp_path)
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "shot_id": "sh-001",
            "feedback": "make it more painterly, keep the warm light",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/keyframe-feedback",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # Prompt was revised and keyframe was regenerated by the self-started storyboard run.
    assert updated.path("storyboard", "keyframes", "sh-001.png").is_file(), (
        "keyframe should have been regenerated by the self-started run"
    )
    assert updated.stage_status("keyframes") == "complete", (
        "keyframes stage should be complete after self-started run (not pending)"
    )
    assert "Click Resume" not in page


def test_http_refresh_knowledge_self_starts_run(tmp_path):
    """After refresh-knowledge POST, the affected stage self-starts without a Resume."""
    p = _storyboard_project(tmp_path)
    # Write the knowledge packet that will be archived.
    packet_path = p.path("knowledge", "packets", "shot-sh-001.json")
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps({
        "purpose": "shot",
        "target": "sh-001",
        "selected_entries": [],
    }))
    p.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({
            "purpose": "shot",
            "target": "sh-001",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/refresh-knowledge",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        page = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    updated = Project.load(tmp_path / p.project_id)
    # Refreshing a shot's knowledge re-plans the storyboard stage; the self-started run
    # re-runs it to 'complete' and invalidates the now-separate keyframes stage to 'pending'
    # (keyframe images are regenerated by the keyframes stage at its own gate, not here).
    assert updated.stage_status("storyboard") == "complete", (
        "storyboard stage should have re-run to complete after the self-started run"
    )
    assert updated.stage_status("keyframes") == "pending", (
        "the downstream keyframes stage should be invalidated to pending"
    )
    assert "Click Resume" not in page


# Task 5: approving a stage auto-runs the next stage to its gate
# ---------------------------------------------------------------

def _fake_project_paused_at_first_gate(tmp_path):
    """Project with storyboard stage at 'complete' (its gate) and video stage pending."""
    from studio_agent import cli
    p = Project.create(
        "approve auto advance",
        root=tmp_path,
        stages=["storyboard", "video"],
        model_config=cli.load_config()["profiles"]["fake"],
    )
    # Set storyboard to complete (at its gate), video stays pending.
    p.set_stage_status("storyboard", "complete")
    p.current_stage = "storyboard"
    p.status = "in_progress"
    # Minimal storyboard fixtures so the video stage can run after approval.
    p.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "camera": "close-up",
        "action": "A student clenches a glowing fist.",
        "duration_s": 3,
        "keyframe": "sh-001.png",
        "characters": [],
        "reference_seed": 42,
        "keyframe_prompt": "close-up of a glowing fist",
    }]}))
    # A genuinely-complete storyboard has written each shot's keyframe prompt file; the gate's
    # on_approve confirms that batch, so the fixture must include it (not just shots.json).
    p.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(
        "Close-up still of a glowing clenched fist."
    )
    p.path("storyboard", "keyframes", "sh-001.png").write_bytes(b"\x89PNG\r\n")
    p.path("story", "creative_brief.json").write_text(json.dumps({
        "visual_direction": {"camera_language": "patient and intimate"},
        "hard_avoidances": [],
    }))
    p.save()
    # Reload so the returned project reflects the migrated pipeline (storyboard → keyframes →
    # video_prompts → video); approving storyboard now advances to the keyframes stage.
    return Project.load(p.dir)


def test_approve_into_bible_does_not_auto_run(tmp_path):
    """Approving the stage before the bible pauses (no auto-run) so references can be uploaded."""
    from studio_agent import cli
    p = Project.create(
        "approve into bible",
        root=tmp_path,
        stages=["plot", "bible"],
        model_config=cli.load_config()["profiles"]["fake"],
    )
    p.set_stage_status("plot", "complete")
    p.current_stage = "plot"
    p.status = "in_progress"
    p.save()

    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({"action": "approve"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{p.project_id}/approve",
            data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30)  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    reloaded = Project.load(p.dir)
    assert reloaded.stage_status("plot") == "approved"
    assert reloaded.current_stage == "bible"
    assert reloaded.stage_status("bible") == "pending", (
        "bible must not auto-run; it waits for the user to upload references and press Resume"
    )


def test_approve_auto_runs_next_stage(tmp_path):
    """A single approve POST leaves the approved stage 'approved' and runs the next stage."""
    project = _fake_project_paused_at_first_gate(tmp_path)
    first, second = project.stages[0], project.stages[1]
    assert project.stage_status(first) == "complete"
    assert project.stage_status(second) == "pending"

    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({"action": "approve"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/approve",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30)  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    reloaded = Project.load(project.dir)
    assert reloaded.stage_status(first) == "approved", (
        f"first stage should be 'approved' after approve POST, got {reloaded.stage_status(first)!r}"
    )
    assert reloaded.stage_status(second) == "complete", (
        f"second stage should have run and paused at 'complete', got {reloaded.stage_status(second)!r}"
    )


def test_approve_final_stage_does_not_run(tmp_path):
    """Approving the final stage marks the project done without trying to run further."""
    from studio_agent import cli
    # Use a genuinely terminal stage: the storyboard→keyframes migration always inserts a
    # keyframes stage after storyboard, so a single-"storyboard" project is no longer final.
    project = Project.create(
        "approve final stage",
        root=tmp_path,
        stages=["assemble"],
        model_config=cli.load_config()["profiles"]["fake"],
    )
    project.set_stage_status("assemble", "complete")
    project.current_stage = "assemble"
    project.status = "in_progress"
    project.save()

    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({"action": "approve"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/approve",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30)  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    reloaded = Project.load(project.dir)
    assert reloaded.stage_status("assemble") == "approved"
    assert reloaded.status == "done"


def test_approve_redirects_without_blocking(tmp_path):
    """approve dispatches the next stage in the background and 303-redirects (no inline render)."""
    project = _fake_project_paused_at_first_gate(tmp_path)
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({"action": "approve"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/approve",
            data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            resp = opener.open(req, timeout=30)  # noqa: S310
            status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert status == 303, f"approve should 303-redirect, got {status}"


def test_self_start_handlers_redirect(tmp_path):
    """regenerate-asset (keyframe/shot_video) and keyframe-feedback dispatch in the background."""
    project = _project_with_shot(tmp_path, qc={"verdict": "pass", "scores": {}})
    # shot_video regeneration re-confirms the compiled video prompt, so it must exist on disk.
    project.path("storyboard", "prompts", "sh-001.video.md").write_text("Close-up: fist tightens.")
    # storyboard stage so keyframe regen is valid
    project.set_stage_status("video", "complete")
    project.current_stage = "video"
    project.save()
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        data = urllib.parse.urlencode({
            "action": "regenerate-asset", "kind": "shot_video", "shot_id": "sh-001",
            "rel_path": "assets/clips/sh-001.mp4",
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/regenerate-asset",
            data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            status = opener.open(req, timeout=30).status  # noqa: S310
        except urllib.error.HTTPError as exc:
            status = exc.code
    finally:
        server.shutdown()
        thread.join(timeout=30)

    assert status == 303, f"shot_video regen should 303-redirect, got {status}"


def test_render_project_shows_style_feedback_form(tmp_path):
    from studio_agent.orchestrator.project import Project
    p = Project.create("style view", root=tmp_path, stages=["style", "bible"])
    p.current_stage = "style"
    p.set_stage_status("style", "complete")
    p.path("bible", "style.md").write_text("# Visual Style\n\nmoody warm look\n")
    p.path("bible", "style_sample.png").write_bytes(b"\x89PNG\r\n")
    p.save()
    html = render_project(p, lang="en")
    assert "style-feedback" in html
    assert "style_sample.png" in html
    assert "moody warm look" in html


def test_style_feedback_accumulates_and_regenerates(tmp_path):
    from studio_agent import cli
    from studio_agent.orchestrator.project import Project
    project = Project.create(
        "style fb", root=tmp_path, stages=["style", "bible"],
        model_config={**cli.load_config()["profiles"]["fake"],
                      "style_name": "custom", "style": {"look": "moody"},
                      "style_label": "Moody", "style_input": {"description": "moody analog"}},
    )
    project.current_stage = "style"
    project.set_stage_status("style", "complete")
    project.path("bible", "style.md").write_text("# Visual Style\n\nmoody\n")
    project.path("bible", "style_sample.png").write_bytes(b"\x89PNG\r\n")
    project.save()

    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({"action": "style-feedback", "feedback": "make it warmer"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/style-feedback",
            data=data, method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        urllib.request.urlopen(req, timeout=30)  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)

    reloaded = Project.load(project.dir)
    assert "make it warmer" in (reloaded.model_config["style_input"]["feedback"])
    assert "warmer" in reloaded.path("bible", "style.md").read_text().lower()
    assert reloaded.stage_status("style") == "complete"  # re-gated, NOT auto-approved
    assert any(d.name.endswith("style-feedback") or "style-feedback" in d.name
               for d in (reloaded.dir / "history").iterdir()) if (reloaded.dir / "history").is_dir() else True


def test_identity_board_html_expands_dict_face_into_rows():
    from studio_agent.web import _identity_board_html

    data = {}
    board = {
        "canonical_face": {"face_shape": "长窄脸型", "eyes": "单眼皮，眼型细长"},
        "canonical_body": "约175cm，偏瘦",
    }
    html = _identity_board_html(data, board)
    assert "{" not in html and "}" not in html
    assert "长窄脸型" in html
    assert "Face shape" in html  # dict sub-field becomes its own labeled row
    assert "约175cm，偏瘦" in html


def _bible_project_with_text_ahead(tmp_path):
    """Build a bible-stage project with character 'gull' whose text is newer than its image."""
    import os
    project = Project.create(
        "seagull story",
        root=tmp_path,
        stages=["plot", "bible"],
    )
    project.set_stage_status("bible", "complete")
    project.current_stage = "bible"
    project.path("story", "plot.json").write_text(json.dumps({
        "logline": "A lonely lighthouse keeper meets a talking gull.",
        "synopsis": "The keeper and the gull become unlikely friends.",
        "themes": ["loneliness"],
    }))
    # Create character 'gull'
    char_dir = project.path("bible", "characters", "gull")
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "character.json").write_text(json.dumps({
        "name": "Gull",
        "description": "A wise old seagull with one broken wing.",
    }))
    (char_dir / "identity_board.json").write_text("{}")
    # Write a reference.png with mtime OLDER than character.json
    ref_png = char_dir / "reference.png"
    ref_png.write_bytes(b"\x89PNG\r\n")
    char_json_mtime = (char_dir / "character.json").stat().st_mtime
    old_mtime = char_json_mtime - 100  # 100 seconds before text was written
    os.utime(ref_png, (old_mtime, old_mtime))
    project.save()
    return project


def test_bible_workbench_renders_inline_controls_and_banner(tmp_path):
    project = _bible_project_with_text_ahead(tmp_path)  # character "gull", text_ahead True
    from studio_agent.web import render_stage_workbench

    html = render_stage_workbench(project, "bible", lang="en")

    # Inline per-card controls present.
    assert "/regenerate-bible-text" in html
    assert "Regenerate text" in html
    # text_ahead True → banner + promoted approve button.
    assert "Text changed since last image" in html
    assert "Approve text &amp; regenerate image" in html
    # Shared sidebar revision/regenerate boxes are suppressed on the bible stage.
    assert "/adjust-artifact" not in html


def _bible_project_with_fake_llm_profile(tmp_path):
    """Build a bible-stage project with a fake LLM profile and character 'gull'."""
    from studio_agent import cli
    project = Project.create(
        "gull story",
        root=tmp_path,
        stages=["plot", "bible", "storyboard"],
        model_config=cli.load_config()["profiles"]["fake"],
    )
    project.set_stage_status("bible", "complete")
    project.current_stage = "bible"
    project.status = "in_progress"
    project.path("story", "plot.json").write_text(json.dumps({
        "logline": "A lonely lighthouse keeper meets a talking gull.",
        "synopsis": "The keeper and the gull become unlikely friends.",
        "themes": ["loneliness"],
    }))
    char_dir = project.path("bible", "characters", "gull")
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / "character.json").write_text(json.dumps({
        "name": "Gull",
        "description": "A wise old seagull with one broken wing.",
    }))
    (char_dir / "identity_board.json").write_text("{}")
    (char_dir / "reference.png").write_bytes(b"\x89PNG\r\n")
    (char_dir / "states.json").write_text("{}")
    project.save()
    return project


def test_regenerate_bible_text_post_revises_without_touching_image(tmp_path):
    project = _bible_project_with_fake_llm_profile(tmp_path)
    ref = project.path("bible", "characters", "gull", "reference.png")
    char_json = project.path("bible", "characters", "gull", "character.json")
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode(
            {"kind": "character", "slug": "gull", "instruction": "make her older"}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/regenerate-bible-text",
            data=data, method="POST",
        )
        urllib.request.urlopen(req, timeout=30)  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)
    assert ref.is_file()  # image never archived by a text regeneration
    # The fake LLM stamps "revision_note" only when apply_artifact_revision ran;
    # without the handler this key is absent, proving the handler actually executed.
    revised = json.loads(char_json.read_text())
    assert "revision_note" in revised, (
        "character.json was not revised by the handler — 'revision_note' key missing"
    )


def test_regenerate_bible_text_post_rejects_empty_comment(tmp_path):
    project = _bible_project_with_fake_llm_profile(tmp_path)
    char_json = project.path("bible", "characters", "gull", "character.json")
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        data = urllib.parse.urlencode({"kind": "character", "slug": "gull", "instruction": "  "}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/regenerate-bible-text",
            data=data, method="POST",
        )
        body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)
    # Assert on the exact validation notice the handler emits, not the generic word "comment"
    # which already appears in the page's comment-box label/placeholder.
    assert "Add a comment before regenerating the text" in body, (
        "Expected exact validation notice in response body"
    )
    # Empty-comment path must NOT have invoked apply_artifact_revision.
    not_revised = json.loads(char_json.read_text())
    assert "revision_note" not in not_revised, (
        "character.json was incorrectly revised despite empty instruction"
    )


def test_keyframes_workbench_shows_structural_controls(tmp_path):
    # Delete + add-shot live on the keyframe review cards (both modes); the per-clip
    # duration control moved to the clip gate's prompt review.
    from studio_agent.web import _render_workbench_shots
    qc = {"overall_pass": True, "summary": "ok"}
    shots = [
        {"id": "s01", "scene": 1, "duration_s": 12, "keyframe_rel": "storyboard/keyframes/s01.png"},
        {"id": "s02", "scene": 1, "duration_s": None, "keyframe_rel": ""},
    ]
    html = _render_workbench_shots(_project_with_shot(tmp_path, qc=qc), shots, "keyframes")
    assert "/delete-clip" in html
    assert "/add-shot" in html
    # A null duration renders as the model-default label, not "None s".
    assert "None s" not in html
    # The last remaining shot cannot be deleted, but adding stays possible.
    single = _render_workbench_shots(_project_with_shot(tmp_path, qc=qc), shots[:1], "keyframes")
    assert "/delete-clip" not in single
    assert "/add-shot" in single
    # the storyboard kind has no structural clip controls on its cards
    story_html = _render_workbench_shots(_project_with_shot(tmp_path, qc=qc), shots, "storyboard")
    assert "/delete-clip" not in story_html
    assert "/set-clip-duration" not in story_html


def test_bible_detail_popup_shows_current_structured_description(tmp_path):
    """Regression: revised text in a structured character.json must be visible.

    With a structured (dict) description, the old card face suppressed the free-text
    overview and showed only the derived identity board, so a text revision was invisible.
    The detail popup renders character.json directly, so the revised token is present.
    """
    from studio_agent.web import render_stage_workbench

    project = Project.create("popup detail", root=tmp_path, stages=["plot", "bible"])
    project.set_stage_status("bible", "complete")
    project.current_stage = "bible"
    cdir = project.path("bible", "characters", "mara")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "character.json").write_text(json.dumps({
        "name": "Mara",
        "description": {"canonical_face": "weathered face with a UNIQUEREVISEDSCAR over the left brow"},
    }, ensure_ascii=False))
    (cdir / "identity_board.json").write_text("{}")
    (cdir / "reference.png").write_bytes(PNG_BYTES)
    project.save()

    html = render_stage_workbench(project, "bible", lang="en")

    # The View-details trigger + its dialog exist...
    assert "data-bible-detail-open='bible-detail-character-mara'" in html
    assert "id='bible-detail-character-mara'" in html
    # ...and the revised, structured description token is rendered inside the page.
    assert "UNIQUEREVISEDSCAR" in html
    # Raw-file link present in the dialog footer.
    assert "artifact?path=bible%2Fcharacters%2Fmara%2Fcharacter.json" in html


def test_bible_card_face_keeps_image_and_controls(tmp_path):
    """Card face keeps the reference image + comment/regenerate controls; detail link present."""
    from studio_agent.web import render_stage_workbench

    project = _bible_project_with_text_ahead(tmp_path)  # character 'gull'
    html = render_stage_workbench(project, "bible", lang="en")

    assert "<figcaption>Identity reference</figcaption>" in html
    assert "View details" in html                             # popup trigger
    assert "/regenerate-bible-text" in html                   # comment + regen controls kept
    assert "Regenerate text" in html


def test_bible_location_detail_popup_has_fields_and_raw_link(tmp_path):
    from studio_agent.web import render_stage_workbench

    project = Project.create("loc popup", root=tmp_path, stages=["plot", "bible"])
    project.set_stage_status("bible", "complete")
    project.current_stage = "bible"
    ldir = project.path("bible", "locations", "lighthouse")
    ldir.mkdir(parents=True, exist_ok=True)
    (ldir / "location.json").write_text(json.dumps({
        "name": "Lighthouse",
        "description": "A salt-bleached tower with a UNIQUELOCTOKEN gallery rail.",
    }, ensure_ascii=False))
    (ldir / "reference.png").write_bytes(PNG_BYTES)
    project.save()

    html = render_stage_workbench(project, "bible", lang="en")

    assert "id='bible-detail-location-lighthouse'" in html
    assert "UNIQUELOCTOKEN" in html
    assert "artifact?path=bible%2Flocations%2Flighthouse%2Flocation.json" in html


def test_bible_card_css_present(tmp_path):
    """The dashboard stylesheet defines the large-card + detail-dialog rules."""
    from studio_agent.web import CSS

    for token in (".bible-card", ".bible-media-row", ".bible-detail-dialog", ".bible-details-link"):
        assert token in CSS


def _prompt_gate_web_project(tmp_path, *, prompt="STATIC EXACT PROMPT"):
    project = Project.create(
        "web prompt gate",
        root=tmp_path,
        stages=["clip", "keyframes", "video_prompts", "video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )
    project.path("storyboard", "shots.json").write_text(json.dumps({"shots": [{
        "id": "sh-001",
        "scene": 1,
        "keyframe": "sh-001.png",
        "camera": "medium shot",
        "camera_movement": "slow push-in",
        "action": "Mara repairs a clockwork bird",
        "start_frame": "Mara holds a still pose beside the clockwork bird",
        "description": "Warm workshop at dusk",
        "characters": [],
        "reference_images": [],
        "reference_seed": 7,
        "duration_s": 4,
    }]}))
    project.path("storyboard", "prompts", "sh-001.keyframe.md").write_text(prompt)
    # Resolve the creative-direction decision the keyframes stage preflight needs, so the
    # self-started keyframe generation runs instead of pausing to ask about camera language.
    project.path("story", "creative_brief.json").write_text(json.dumps({
        "visual_direction": {"camera_language": "patient and intimate"},
        "hard_avoidances": [],
    }))
    project.set_stage_status("clip", "complete")
    project.current_stage = "clip"
    project.save()
    return project


def _post_prompt_action(tmp_path, project, action, fields):
    jobs = JobRunner(inline=True)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, jobs=jobs))
    except PermissionError:
        pytest.skip("local socket bind is unavailable in this sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/projects/{project.project_id}/{action}",
            data=urllib.parse.urlencode(fields).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return urllib.request.urlopen(request, timeout=30).read().decode("utf-8")  # noqa: S310
    finally:
        server.shutdown()
        thread.join(timeout=30)


def test_prompt_chat_post_revises_selected_without_dispatching_media(tmp_path):
    project = _prompt_gate_web_project(tmp_path)

    response = _post_prompt_action(tmp_path, project, "prompt-chat", {
        "gate": "keyframes",
        "shot_id": "sh-001",
        "message": "move her left",
    })

    assert "move her left" in response
    assert project.path("storyboard", "prompts", "sh-001.keyframe.md").is_file()
    assert not project.path("storyboard", "keyframes", "sh-001.png").exists()


def test_confirm_prompts_hashes_all_and_dispatches_keyframes(tmp_path):
    project = _prompt_gate_web_project(tmp_path)

    _post_prompt_action(
        tmp_path, project, "confirm-prompts", {"kind": "keyframes"}
    )

    updated = Project.load(project.dir)
    assert updated.path("storyboard", "prompt_approvals", "keyframes.json").is_file()
    assert updated.path("storyboard", "keyframes", "sh-001.png").is_file()


def test_confirm_prompts_trusts_plain_keyframe_motion_language(tmp_path):
    # Keyframe prompts are no longer prose-validated: a still that happens to mention camera or
    # temporal wording confirms without error (the director is trusted and a human reviews the
    # prompt at the gate). Only motion GRIDS keep a structural panel-enumeration check.
    from studio_agent.prompt_approvals import (
        confirm_prompt_batch,
        require_prompt_batch_approval,
    )

    project = _prompt_gate_web_project(
        tmp_path, prompt="then the camera pushes in on her trembling face"
    )

    manifest = confirm_prompt_batch(project, "keyframes", confirmer="human")

    assert "sh-001" in manifest["prompts"]
    require_prompt_batch_approval(project, "keyframes")  # does not raise


def test_prompt_review_explains_why_confirm_disabled_for_offscreen_invalid_shots(tmp_path):
    """A disabled 'Confirm all' must say WHICH shots block it, not just go dead.

    Regression: with many shots and only a few invalid, batch_ready is False so the confirm
    button renders disabled, but the errors were only shown for the selected shot (shots[0])
    and buried far down the page. A user staring at shot sh-001 (valid) sees a dead button
    and no reason — 'clicking confirm does nothing'. The rail must surface the blocking shots
    with jump links right next to the confirm control.
    """
    from studio_agent.web import _render_prompt_review

    project = _prompt_gate_web_project(tmp_path)
    payload = {
        "gate": "keyframes",
        "batch_ready": False,
        "shots": [
            {"id": "sh-001", "prompt_status": "ready", "validation_errors": [],
             "prompt_text": "still portrait", "prompt_kind": "static",
             "prompt_rel": "storyboard/prompts/sh-001.keyframe.md"},
            {"id": "sh-023", "prompt_status": "invalid", "validation_errors": ["camera movement"],
             "prompt_text": "the camera pushes in", "prompt_kind": "static",
             "prompt_rel": "storyboard/prompts/sh-023.keyframe.md"},
        ],
    }

    html = _render_prompt_review(project, payload, can_confirm=True)

    # Confirm stays disabled while any prompt is invalid...
    assert "disabled" in html
    # ...and the rail now names the blocking shot with a jump link and its reason.
    assert "prompt-blockers" in html
    assert "#prompt-sh-023" in html
    assert "camera movement" in html


def test_reference_picker_accumulates_multiple_files(tmp_path):
    """The reference-photo picker must let the user build up MANY references.

    Regression: the change handler used to replace the running selection on every
    pick (`selectedFiles = Array.from(input.files)`) and never push the combined list
    back onto the input, so choosing files one at a time left exactly one. The handler
    must instead append newly picked files to `selectedFiles` and call syncFileInput().
    """
    from studio_agent.web import BUSY_JS

    # The replace-on-every-pick assignment is gone...
    assert "selectedFiles = Array.from(input.files" not in BUSY_JS
    # ...replaced by accumulation that pushes the combined list back to the input.
    assert "selectedFiles.push(" in BUSY_JS
    assert "syncFileInput();" in BUSY_JS


def test_bible_detail_dialog_js_registered(tmp_path):
    from studio_agent.web import BUSY_JS

    assert "function studioInitBibleDetailDialogs()" in BUSY_JS
    assert "studioInitBibleDetailDialogs();" in BUSY_JS
    assert "data-bible-detail-open" in BUSY_JS


def test_bible_popup_strings_translated_to_zh(tmp_path):
    from studio_agent.i18n import t

    assert t("View details", "zh") != "View details"
    assert t("Open raw file", "zh") != "Open raw file"
    assert t("Close", "zh") != "Close"


def test_clip_prompt_review_shows_structure_and_duration_controls(tmp_path):
    from studio_agent.web import _render_prompt_review

    clip_stages = [
        "style", "concept", "bible", "clip", "keyframes", "video_prompts",
        "video", "audio", "assemble",
    ]
    p = Project.create("clip prompt gate", root=tmp_path, stages=clip_stages)
    payload = {
        "gate": "keyframes",
        "batch_ready": True,
        "cost": 0.0,
        "cost_cap": None,
        "shots": [
            {"id": "sh-001", "prompt_status": "ready", "prompt_text": "a", "duration_s": 6,
             "prompt_rel": "storyboard/prompts/sh-001.keyframe.md"},
            {"id": "sh-002", "prompt_status": "ready", "prompt_text": "b", "duration_s": None,
             "prompt_rel": "storyboard/prompts/sh-002.keyframe.md"},
        ],
    }
    html = _render_prompt_review(p, payload, can_confirm=True)
    # Structural edits at the clip gate: add (incl. at start), delete, per-clip duration.
    assert html.count("/add-shot") >= 3
    assert "/delete-clip" in html
    assert "/set-clip-duration" in html
    # Duration bounds come from capabilities (defaults here), not hardcoded 4-15.
    assert "min='1'" in html and "max='15'" in html

    # The video-prompts gate offers add/delete but no duration editor.
    payload["gate"] = "videos"
    video_html = _render_prompt_review(p, payload, can_confirm=True)
    assert "/add-shot" in video_html and "/delete-clip" in video_html
    assert "/set-clip-duration" not in video_html

    # Story mode's storyboard gate gets add/delete too, still without duration forms.
    story_stages = [
        "style", "plot", "script", "bible", "storyboard", "keyframes",
        "video_prompts", "video", "audio", "assemble",
    ]
    story = Project.create("story prompt gate", root=tmp_path, stages=story_stages)
    payload["gate"] = "keyframes"
    story_html = _render_prompt_review(story, payload, can_confirm=True)
    assert "/add-shot" in story_html and "/delete-clip" in story_html
    assert "/set-clip-duration" not in story_html


def test_dashboard_clip_plan_fields_hidden_unless_clip_format_is_default(tmp_path):
    from studio_agent import cli
    from studio_agent.web import CSS, JobRunner, _render_dashboard

    config = cli.load_config()
    jobs = JobRunner(inline=True)

    # Story default (short_film): clip-plan block hidden, Length visible.
    story_html = _render_dashboard(tmp_path, config, jobs)
    assert "data-clip-plan hidden" in story_html
    assert "data-length-label hidden" not in story_html

    # Clip default: clip-plan block visible, Length hidden.
    clip_config = dict(config)
    clip_config["default_format"] = "short_video"
    clip_html = _render_dashboard(tmp_path, clip_config, jobs)
    assert "data-clip-plan hidden" not in clip_html
    assert "data-clip-plan>" in clip_html
    assert "data-length-label hidden" in clip_html

    # label/.model-mix set display:grid, which beats the UA [hidden] rule — the theme
    # must restate display:none for these toggled elements or they render regardless.
    assert "[data-clip-plan][hidden], [data-length-label][hidden] { display:none; }" in CSS


def test_apply_video_model_swaps_only_the_video_component():
    from studio_agent.web import apply_video_model
    from studio_agent import cli

    config = cli.load_config()
    model_config = {
        "llm": "fake",
        "image": "fake",
        "video": "byteplus",
        "video_model": "OLD-MODEL",
        "video_stale_only_key": "leftover",
        "model_parts": {"llm": "l1", "image": "i1", "video": "byteplus-seedance-fast"},
        "style": {"look": "cartoon"},
    }

    out = apply_video_model(config, model_config, "gemini-veo-3-1")

    veo = config["model_options"]["video"]["gemini-veo-3-1"]
    # Video component is now the chosen model...
    assert out["video"] == veo["video"]
    assert out["video_model"] == veo["video_model"]
    # ...stale video key from the old model is dropped...
    assert "video_stale_only_key" not in out
    # ...model_parts.video updated, other parts preserved...
    assert out["model_parts"]["video"] == "gemini-veo-3-1"
    assert out["model_parts"]["llm"] == "l1"
    # ...and non-video config is byte-identical.
    assert out["llm"] == "fake"
    assert out["image"] == "fake"
    assert out["style"] == {"look": "cartoon"}


def test_apply_video_model_rejects_unknown_choice():
    from studio_agent.web import apply_video_model
    from studio_agent import cli
    import pytest

    with pytest.raises(ValueError):
        apply_video_model(cli.load_config(), {"video": "fake"}, "not-a-real-model")


def test_video_model_key_missing_flags_unset_env(monkeypatch):
    from studio_agent.web import video_model_key_missing
    from studio_agent import cli

    config = cli.load_config()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert video_model_key_missing(config, "gemini-veo-3-1") == "GEMINI_API_KEY"
    monkeypatch.setenv("GEMINI_API_KEY", "present")
    assert video_model_key_missing(config, "gemini-veo-3-1") is None


def test_video_model_key_missing_none_for_fake():
    from studio_agent.web import video_model_key_missing
    from studio_agent import cli

    assert video_model_key_missing(cli.load_config(), "fake") is None


def test_video_model_key_missing_flags_volcengine_default_without_key(monkeypatch):
    from studio_agent.web import video_model_key_missing
    from studio_agent import cli

    config = cli.load_config()
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    # china-seedance-fast (provider volcengine) declares its key via the provider
    # fallback map, not video_api_key_env — the guardrail must still catch it.
    assert video_model_key_missing(config, "china-seedance-fast") == "ARK_API_KEY"
    monkeypatch.setenv("ARK_API_KEY", "present")
    assert video_model_key_missing(config, "china-seedance-fast") is None


def test_current_video_model_prefers_model_parts(tmp_path):
    from studio_agent.web import current_video_model
    from studio_agent import cli

    config = cli.load_config()
    mc = {"model_parts": {"video": "gemini-veo-3-1"}, "video": "gemini-veo"}
    assert current_video_model(config, mc) == "gemini-veo-3-1"
    # Falls back to the config default when nothing usable is present.
    assert current_video_model(config, {}) == config["default_model_options"]["video"]


def test_maybe_apply_video_model_persists_new_choice(tmp_path, monkeypatch):
    from studio_agent.web import maybe_apply_video_model, current_video_model
    from studio_agent import cli
    from studio_agent.orchestrator.project import Project

    monkeypatch.setenv("GEMINI_API_KEY", "present")
    config = cli.load_config()
    project = Project.create(
        "switch video", root=tmp_path, stages=["video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )

    notice = maybe_apply_video_model(project, config, {"video_model": ["gemini-veo-3-1"]})

    assert notice is None
    # Reload from disk to prove it persisted. Project.load takes the project DIRECTORY.
    reloaded = Project.load(project.dir)
    assert current_video_model(config, reloaded.model_config) == "gemini-veo-3-1"
    assert reloaded.model_config["llm"] == "fake"  # untouched


def test_maybe_apply_video_model_blocks_on_missing_key(tmp_path, monkeypatch):
    from studio_agent.web import maybe_apply_video_model
    from studio_agent import cli
    from studio_agent.orchestrator.project import Project

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = cli.load_config()
    project = Project.create(
        "blocked video", root=tmp_path, stages=["video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake"},
    )

    notice = maybe_apply_video_model(project, config, {"video_model": ["gemini-veo-3-1"]})

    assert notice is not None and "GEMINI_API_KEY" in notice
    # Nothing changed on disk.
    reloaded = Project.load(project.dir)
    assert reloaded.model_config["video"] == "fake"


def test_maybe_apply_video_model_noop_when_unchanged(tmp_path):
    from studio_agent.web import maybe_apply_video_model
    from studio_agent import cli
    from studio_agent.orchestrator.project import Project

    config = cli.load_config()
    project = Project.create(
        "noop video", root=tmp_path, stages=["video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake",
                      "model_parts": {"video": "fake"}},
    )
    assert maybe_apply_video_model(project, config, {"video_model": ["fake"]}) is None
    assert maybe_apply_video_model(project, config, {}) is None


def test_shot_card_shows_video_model_picker(tmp_path):
    from studio_agent.web import _render_shot_card
    from studio_agent.orchestrator.project import Project

    project = Project.create(
        "picker card", root=tmp_path, stages=["video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake",
                      "model_parts": {"video": "fake"}},
    )
    review = {
        "id": "sh-001", "clip": None, "keyframe": None, "duration_s": 4,
        "camera": "", "camera_movement": "", "video_prompt": None, "review_mode": "manual",
    }

    html = _render_shot_card(project, review, lang="en")

    # The regenerate form now carries a video-model select, defaulted to the current model.
    assert "name='video_model'" in html
    assert "Veo 3.1" in html            # a catalog label is present
    assert "value='fake' selected" in html  # current selection preselected


def test_regenerate_applies_selected_video_model(tmp_path, monkeypatch):
    from studio_agent import web
    from studio_agent.orchestrator.project import Project

    monkeypatch.setenv("GEMINI_API_KEY", "present")
    project = Project.create(
        "regen switch", root=tmp_path, stages=["video"],
        model_config={"llm": "fake", "image": "fake", "video": "fake",
                      "model_parts": {"video": "fake"}},
    )
    # A shot must exist so cli._find_shot resolves it.
    import json
    project.path("storyboard", "shots.json").write_text(
        json.dumps({"shots": [{"id": "sh-001", "keyframe": "sh-001.png"}]})
    )
    called = {}
    monkeypatch.setattr(web, "regenerate_shot_video",
                        lambda proj, sid: called.setdefault("shot", sid))

    from studio_agent import cli
    notice = web.maybe_apply_video_model(project, cli.load_config(),
                                         {"video_model": ["gemini-veo-3-1"]})

    assert notice is None
    reloaded = Project.load(project.dir)
    assert web.current_video_model(cli.load_config(), reloaded.model_config) == "gemini-veo-3-1"
