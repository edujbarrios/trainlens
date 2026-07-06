from trainlens.llm.prompts import (
    ReportPromptContext,
    ReportPromptTemplate,
    render_ml_results_explanation_prompt,
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
