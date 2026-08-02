"""LLM-backed style profiler: the universal default for custom styles.

Every run profile has an LLM, and turning a free-text style description into a structured
style guide is fundamentally an LLM task. So when no explicit/vision style profiler is
configured, the LLM serves as the default — this is what unblocks custom styles on the
dashboard model-mix and other real profiles.
"""

from studio_agent import cli
from studio_agent.providers.base import Generation, StyleProfiler
from studio_agent.providers.llm_style import LLMStyleProfiler


class _StubLLM:
    def __init__(self, content):
        self._content = content
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None):  # pragma: no cover - unused
        raise NotImplementedError

    def complete_json(self, prompt, *, system=None):
        self.prompts.append(prompt)
        return Generation(content=self._content, provider="stub-llm", model="stub-1",
                          cost_usd=0.002, seconds=0.05)


def test_llm_style_profiler_normalizes_and_carries_cost():
    llm = _StubLLM({
        "look": "1970s analog sci-fi",
        "label": "70s analog sci-fi",
        "prompt_playbook": "Grainy 35mm anamorphic, amber/teal grade...",
    })
    profiler = LLMStyleProfiler(llm)

    gen = profiler.profile(description="1970s grainy analog sci-fi")

    assert isinstance(profiler, StyleProfiler)
    assert gen.content["look"] == "1970s analog sci-fi"
    assert gen.content["aspect_ratio"] == "16:9"  # backfilled
    assert gen.content["prompt_playbook"].strip()
    assert gen.provider == "stub-llm" and gen.cost_usd == 0.002
    assert "1970s grainy analog sci-fi" in llm.prompts[0]


def test_build_providers_defaults_to_llm_style_profiler_for_model_mix():
    # A dashboard model-mix wires a real LLM but no style_profiler; the default must
    # fill in so custom styles do not hard-fail.
    providers = cli.build_providers({
        "llm": "openai-compatible",
        "llm_base_url": "http://localhost:8000/v1",
        "llm_model": "local-model",
        "image": "fake",
        "video": "fake",
    })

    assert isinstance(providers.style_profiler, LLMStyleProfiler)


def test_fake_profile_still_uses_the_fake_style_profiler():
    from studio_agent.providers.fake import FakeStyleProfiler

    providers = cli.build_providers({"llm": "fake"})

    assert isinstance(providers.style_profiler, FakeStyleProfiler)


def test_llm_prompt_requests_the_new_style_dimensions():
    llm = _StubLLM({"look": "x"})
    LLMStyleProfiler(llm).profile(description="anything")
    prompt = llm.prompts[-1]
    for key in ("medium", "idiom", "lighting", "color_grade", "lens", "atmosphere"):
        assert key in prompt
