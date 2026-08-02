# Studio Agent

Studio Agent is an AI filmmaking system for turning one idea into a complete AI video or TV
episode. It orchestrates the full creative pipeline from concept and script through show bible,
storyboard, image prompts, keyframes, video prompts, generated clips, audio, review gates, and final
assembly.

This is a real product in active development, not a demo wrapper around a single model. The core
focus is long-form consistency, human control, provider flexibility, and file-backed editability so a
creator can intervene at any point and regenerate only the part that needs to change.

## Product Direction

Studio Agent is being built around a simple but hard goal:

> Give the system an idea, review or revise the important creative decisions, and get back an
> editable finished video.

The long-term target is full AI video and TV-series production. The current build already runs the
core pipeline end-to-end and can produce assembled video from an idea. Active work is focused on
making longer, multi-scene outputs more reliable: stronger character consistency, better prompt
adaptation per provider, cleaner audio, faster review loops, and more predictable regeneration.

## What It Does

- Expands a raw idea into plot, script, and a reusable show bible.
- Creates character identity boards and location references to preserve visual continuity.
- Compiles editable image and video prompts from structured story, shot, bible, style, and
  provider-capability data.
- Requires human review gates by default before expensive or irreversible steps.
- Runs end-to-end in offline fake mode for local development and testing.
- Supports real provider integrations behind narrow interfaces, including image, video, LLM, VLM,
  and audio paths, so the same pipeline can produce real media when configured with provider keys.
- Saves project state as plain files under `projects/<project-id>/` so work can be inspected,
  edited, resumed, and regenerated locally.
- Supports English and Chinese projects as first-class workflows.

## Current Status

Studio Agent is functional end-to-end today. It can take an idea through planning, bible creation,
storyboarding, prompt generation, keyframes, video clips, audio, and final assembly into a playable
MP4.

The product is still under active development. There are bugs, provider-specific edge cases, and
quality issues being worked through, especially around longer multi-scene runs, consistency across
shots, and the polish expected from a production creative tool.

The repository is updated continuously as the product evolves, with the public surface focused on the
current product architecture and runnable code.

## Architecture

Studio Agent is intentionally file-first and provider-agnostic.

```text
idea
  -> style
  -> plot / concept
  -> script
  -> show bible
  -> storyboard
  -> image prompt review
  -> keyframes
  -> video prompt review
  -> video clips
  -> audio
  -> assembly
  -> final MP4
```

Key modules:

- `studio_agent/orchestrator/` - project state, resumability, and stage coordination.
- `studio_agent/stages/` - deterministic pipeline stages.
- `studio_agent/providers/` - model/provider adapters behind shared interfaces.
- `studio_agent/assembly/` - FFmpeg-based final video assembly.
- `studio_agent/web.py` - local browser dashboard for review and regeneration.
- `skills/` and `knowledge/` - local filmmaking and prompting knowledge used by the runtime.
- `tests/` - fake-provider and deterministic behavior coverage.

## Quickstart

Use Python 3.10+; Python 3.11 is preferred. FFmpeg is required for final MP4 assembly.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the offline fake-provider pipeline, which exercises the same file-backed orchestration without
API keys or paid calls:

```bash
python -m studio_agent.cli run \
  --idea "a lonely lighthouse keeper meets a mysterious signal in the fog" \
  --fake \
  --auto
```

Start the local review dashboard:

```bash
python -m studio_agent.cli web --port 8765
```

Run deterministic tests:

```bash
python -m pytest
```

## Real Providers

Fake mode requires no keys and is the safest way to inspect the product architecture. Real-provider
runs use the same pipeline with configured model profiles and local API keys.

Real-provider profiles are wired for image, video, LLM, and audio experiments. API keys must be
stored locally in `.env`, which is ignored by Git. Copy `.env.example` to `.env` and fill in only the
providers you intend to test.

```bash
cp .env.example .env
```

Never commit `.env`, generated project outputs, provider responses, or paid media artifacts.

## Security And Privacy

This repository is configured so local secrets and generated outputs are not tracked:

- `.env` is ignored.
- `projects/` is ignored.
- generated media and project artifacts should stay local unless intentionally published elsewhere.

If you discover a sensitive-data exposure risk, please open a private channel with the maintainer
instead of posting credentials or exploit details publicly.

## Repository Notice

This repository is public for transparency, technical review, and product development visibility.
No open-source license has been granted at this time. Unless a license file is added later, all
rights are reserved by the author.
