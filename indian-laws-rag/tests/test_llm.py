from app.generation.llm import ANSWER_MODELS, UTILITY_MODELS, model_for


def test_every_provider_has_both_model_tiers():
    assert set(ANSWER_MODELS) == set(UTILITY_MODELS)


def test_the_answer_tier_is_the_default():
    assert model_for("openai") == ANSWER_MODELS["openai"]


def test_the_utility_tier_is_selected_by_purpose():
    assert model_for("openai", "utility") == UTILITY_MODELS["openai"]


def test_the_utility_model_differs_from_the_answer_model():
    # Otherwise the split saves nothing.
    assert UTILITY_MODELS["openai"] != ANSWER_MODELS["openai"]


def test_an_environment_variable_overrides_the_answer_model(monkeypatch):
    monkeypatch.setenv("OPENAI_ANSWER_MODEL", "gpt-4.1")

    assert model_for("openai", "answer") == "gpt-4.1"


def test_an_environment_variable_overrides_the_utility_model(monkeypatch):
    monkeypatch.setenv("GROQ_UTILITY_MODEL", "custom-small")

    assert model_for("groq", "utility") == "custom-small"


def test_an_override_for_one_purpose_leaves_the_other_alone(monkeypatch):
    monkeypatch.setenv("OPENAI_UTILITY_MODEL", "custom-nano")

    assert model_for("openai", "utility") == "custom-nano"
    assert model_for("openai", "answer") == ANSWER_MODELS["openai"]
