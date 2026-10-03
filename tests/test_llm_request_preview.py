from trainlens import OpenAICompatibleProvider, preview_llm_request
from trainlens.prompt_recipes import improvement_plan_prompt


def test_preview_llm_request_combines_prompt_and_sanitized_notebook_evidence() -> None:
    namespace = {
        "history": {
            "train_loss": [0.9, 0.6, 0.4],
            "val_loss": [0.95, 0.7, 0.72],
        }
    }
    provider = OpenAICompatibleProvider.from_values(
        base_url="http://localhost:11434/v1",
        model="local-model",
    )
    prompt = improvement_plan_prompt(
        objective="Find the safest next experiment for the validation regression."
    )

    preview = preview_llm_request(
        namespace,
        mode="improvement_ideas",
        provider=provider,
        prompt_options=prompt,
    )

    assert preview.endpoint == "http://localhost:11434/v1/chat/completions"
    assert "Find the safest next experiment" in preview.system_prompt
    assert "evidence, proposed change, rationale, expected effect" in preview.system_prompt
    assert "validation_loss" in preview.user_prompt
    assert "Deterministic TrainLens Findings" in preview.user_prompt


def test_preview_llm_request_can_resolve_provider_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TRAINLENS_LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("TRAINLENS_LLM_MODEL", "env-model")

    preview = preview_llm_request(
        {"metrics": {"accuracy": 0.91}},
        mode="improvement_ideas",
    )

    assert preview.model == "env-model"
    assert preview.endpoint == "https://api.example.com/v1/chat/completions"
