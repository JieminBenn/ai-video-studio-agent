"""Deterministic bilingual checks and a human A/B worksheet for creative direction.

This harness verifies structural regressions (retrieval coverage, locked constraints,
avoidance encoding, and reproducibility).  It intentionally does not assign a taste or
preference score; the blinded Markdown worksheet is the human evaluation surface.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import yaml

from .knowledge import KnowledgeQuery, load_packaged_core
from .knowledge.retrieval import retrieve


_AB_SEED = 20260622


def _query(
    scenario: dict[str, Any],
    *,
    domain: str,
) -> KnowledgeQuery:
    stage = "image_prompt" if domain == "prompting" else "storyboard"
    return KnowledgeQuery(
        stage=stage,
        domains=(domain,),
        intent_text=" ".join([
            str(scenario["idea"]),
            str(scenario["focus"]),
            *(str(item) for item in scenario["locked_constraints"]),
            *(str(item) for item in scenario["required_facts"]),
        ]),
        intents=(str(scenario["focus"]), "consistency", "continuity", "pacing"),
        style="anime" if scenario["focus"] == "anime" else "",
        format_name="short_film",
        language=str(scenario["language"]),
        capabilities=frozenset({"camera_motion"}),
        hard_avoidances=tuple(str(item) for item in scenario["forbidden_contradictions"]),
    )


def _baseline(scenario: dict[str, Any]) -> str:
    constraints = "; ".join(str(item) for item in scenario["locked_constraints"])
    return f"{scenario['idea']} Style and direction: {constraints}."


def _adaptive(scenario: dict[str, Any], ranked) -> str:
    sections = [
        f"Idea: {scenario['idea']}",
        "Locked constraints: " + "; ".join(scenario["locked_constraints"]),
        "Required prompt facts: " + "; ".join(scenario["required_facts"]),
        "Retrieved filmmaking recipes:",
    ]
    sections.extend(
        f"- [{item.entry.id}] {item.entry.title}: {item.entry.recipe or item.entry.principle}"
        for item in ranked
    )
    sections.append(
        "Hard avoidances: " + "; ".join(scenario["forbidden_contradictions"])
    )
    return "\n".join(sections)


def _scenario_result(scenario: dict[str, Any], entries) -> dict[str, Any]:
    ranked = []
    repeated = []
    for domain in scenario["expected_domains"]:
        query = _query(scenario, domain=str(domain))
        ranked.extend(retrieve(entries, query, limit=2))
        repeated.extend(retrieve(entries, query, limit=2))
    ids = [item.entry.id for item in ranked]
    repeated_ids = [item.entry.id for item in repeated]
    domains = sorted({item.entry.domain for item in ranked})
    adaptive = _adaptive(scenario, ranked)
    checks = {
        "retrieval_domain_coverage": set(scenario["expected_domains"]) <= set(domains),
        "locked_constraints_present": all(
            str(item) in adaptive for item in scenario["locked_constraints"]
        ),
        "required_facts_present": all(
            str(item) in adaptive for item in scenario["required_facts"]
        ),
        "forbidden_contradictions_encoded_as_avoidances": all(
            str(item) in adaptive.split("Hard avoidances:", 1)[-1]
            for item in scenario["forbidden_contradictions"]
        ),
        "retrieval_reproducible": ids == repeated_ids,
    }
    return {
        "id": scenario["id"],
        "language": scenario["language"],
        "focus": scenario["focus"],
        "passed": all(checks.values()),
        "checks": checks,
        "retrieval_ids": ids,
        "retrieval_domains": domains,
        "baseline_artifact": _baseline(scenario),
        "adaptive_artifact": adaptive,
    }


def _worksheet(results: list[dict[str, Any]]) -> str:
    rng = random.Random(_AB_SEED)
    lines = [
        "# Human A/B creative-direction review",
        "",
        f"Blinding seed: `{_AB_SEED}`",
        "",
        "Score each artifact from 1–5. Do not reveal the mapping until all rows are filled.",
        "",
    ]
    for item in results:
        adaptive_is_a = bool(rng.getrandbits(1))
        artifacts = {
            "A": item["adaptive_artifact"] if adaptive_is_a else item["baseline_artifact"],
            "B": item["baseline_artifact"] if adaptive_is_a else item["adaptive_artifact"],
        }
        lines.extend([
            f"## {item['id']}",
            "",
            "| artifact | camera motivation (1–5) | identity specificity (1–5) | prompt usefulness (1–5) |",
            "|---|---:|---:|---:|",
            "| A |  |  |  |",
            "| B |  |  |  |",
            "",
            "### Artifact A",
            "",
            artifacts["A"],
            "",
            "### Artifact B",
            "",
            artifacts["B"],
            "",
            f"<!-- mapping: A={'adaptive' if adaptive_is_a else 'baseline'}, "
            f"B={'baseline' if adaptive_is_a else 'adaptive'} -->",
            "Human result: _unscored_",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def run_eval(*, scenarios_path, out_dir, providers) -> dict[str, Any]:
    """Run deterministic checks and write the JSON report plus blinded worksheet."""
    scenarios_path = Path(scenarios_path)
    out_dir = Path(out_dir)
    scenarios = yaml.safe_load(scenarios_path.read_text()).get("scenarios", [])
    entries = load_packaged_core()
    results = [_scenario_result(dict(scenario), entries) for scenario in scenarios]
    passes = sum(1 for item in results if item["passed"])
    provider = getattr(getattr(providers, "llm", None), "name", "none")
    report = {
        "version": 1,
        "seed": _AB_SEED,
        "provider": provider,
        "summary": {
            "scenarios": len(results),
            "deterministic_passes": passes,
            "failures": len(results) - passes,
        },
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath("eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    out_dir.joinpath("human-ab-review.md").write_text(_worksheet(results))
    return report
