from studio_agent.i18n import LANGS, UI_STRINGS, t, lang_from_cookie_header


def test_t_passthrough_for_english():
    assert t("Projects", "en") == "Projects"


def test_t_translates_known_string_to_chinese():
    assert t("Projects", "zh") == "项目"


def test_t_falls_back_to_english_source_for_unknown_string():
    assert t("Totally new label", "zh") == "Totally new label"


def test_t_strips_whitespace_when_matching():
    # callers pass raw literals; matching is on the trimmed key
    assert t("  Projects  ", "zh") == "项目"


def test_langs_are_en_and_zh():
    assert LANGS == ("en", "zh")


def test_lang_from_cookie_header_reads_studio_lang():
    assert lang_from_cookie_header("studioLang=zh") == "zh"
    assert lang_from_cookie_header("studioLang=en") == "en"


def test_lang_from_cookie_header_defaults_to_en():
    assert lang_from_cookie_header(None) == "en"
    assert lang_from_cookie_header("") == "en"
    assert lang_from_cookie_header("studioLang=fr") == "en"
    assert lang_from_cookie_header("other=1") == "en"


def test_ui_strings_are_nonempty_chinese_catalog():
    assert UI_STRINGS["Resume"] == "继续"
    assert UI_STRINGS["Approve this stage"] == "批准此阶段"
