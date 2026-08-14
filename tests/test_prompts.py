from trainlens.llm.prompts import (
    ReportPromptContext,
    ReportPromptTemplate,
    render_ml_results_explanation_prompt,
    show_trainlens_prompts,
)


def test_report_prompt_template_renders_parameters():
    prompt = ReportPromptTemplate().render(
        ReportPromptContext(
            markdown_report="## TrainLens Report\n\n- eval_loss=1.2",
            llm_model="gpt-test",
            model_family="VLM projector fine-tune",
            audience="research engineers",
            tone="direct",
            rules=("Never invent metrics.",),
            focus_areas=("projector alignment",),
        )
    )

    assert "VLM projector fine-tune" in prompt
    assert "research engineers" in prompt
    assert "Never invent metrics." in prompt
    assert "projector alignment" in prompt
    assert "eval_loss=1.2" in prompt
    assert "LLM model used for this report: gpt-test" in prompt


def test_report_prompt_template_redacts_secrets_before_llm_prompt():
    prompt = ReportPromptTemplate().render(
        ReportPromptContext(
            markdown_report="## TrainLens Report\n\n- api_key=sk-test1234567890",
        )
    )

    assert "sk-test1234567890" not in prompt
    assert "[REDACTED]" in prompt


def test_ml_results_prompt_asks_llm_to_explain_with_context():
    prompt = render_ml_results_explanation_prompt(
        "# TrainLens Notebook Context\n\n- train_loss=0.18\n- eval_loss=0.57",
        llm_model="gpt-paper",
    )

    assert "generate a TrainLens paper-style Markdown report" in prompt
    assert "Notebook context" in prompt
    assert "Explain why the observed result likely happened" in prompt
    assert "## TrainLens Scientific Report" in prompt
    assert "gpt-paper" in prompt


def test_improvement_ideas_prompt_requests_experiment_plan():
    prompt = render_ml_results_explanation_prompt(
        "# TrainLens Notebook Context\n\n- validation_loss=0.57",
        mode="improvement_ideas",
        llm_model="gpt-ideas",
    )

    assert "## TrainLens Improvement Ideas" in prompt
    assert "Improvement Ideas" in prompt
    assert "Prioritized Experiments" in prompt
    assert "gpt-ideas" in prompt


def test_builtin_prompts_are_discoverable_from_public_api():
    from trainlens import show_trainlens_prompts as public_show_prompts

    prompts = public_show_prompts()

    assert prompts == show_trainlens_prompts()
    assert [prompt.name for prompt in prompts] == [
        "scientific_report",
        "improvement_plan",
        "training_diagnosis",
        "experiment_design",
    ]


def test_named_prompt_can_be_fully_parameterized():
    prompt = render_ml_results_explanation_prompt(
        "loss=0.8",
        prompt_name="training_diagnosis",
        objective="Find the most likely reason convergence stopped.",
        heading="## Custom Diagnosis",
        audience="new ML practitioners",
        tone="concise and educational",
        model_family="image classifier",
        rules=("Use only supplied values.",),
        focus_areas=("learning-rate schedule",),
        return_instructions=("Return exactly three ranked hypotheses.",),
    )

    assert "Find the most likely reason convergence stopped." in prompt
    assert "## Custom Diagnosis" in prompt
    assert "new ML practitioners" in prompt
    assert "image classifier" in prompt
    assert "learning-rate schedule" in prompt
    assert "exactly three ranked hypotheses" in prompt


def test_unknown_named_prompt_lists_available_prompts():
    import pytest

    with pytest.raises(ValueError, match="Available: scientific_report"):
        render_ml_results_explanation_prompt("loss=0.8", prompt_name="missing")
