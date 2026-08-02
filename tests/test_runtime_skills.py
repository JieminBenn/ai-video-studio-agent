from studio_agent.runtime_skills import load_prompt_skill, load_prompt_skills


def test_loads_packaged_prompt_skill():
    skill = load_prompt_skill("micro_expression")

    assert "micro-expression" in skill.lower()
    assert "temporal" in skill.lower()


def test_load_prompt_skills_ignores_missing_names():
    skills = load_prompt_skills(["character_identity", "missing_skill"])

    assert "character_identity" in skills
    assert "missing_skill" not in skills
    assert "turnaround" in skills["character_identity"].lower()


def test_story_flow_skill_loads_and_defines_advancing():
    from studio_agent.runtime_skills import load_prompt_skill

    text = load_prompt_skill("story_flow")
    assert text  # file exists and is non-empty
    lowered = text.lower()
    assert "advance" in lowered
    assert "redundant" in lowered
