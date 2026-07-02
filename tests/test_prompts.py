from trainlens.llm.prompts import ReportPromptContext, ReportPromptTemplate


def test_report_prompt_template_renders_parameters():
    prompt = ReportPromptTemplate().render(
        ReportPromptContext(
            markdown_report="## TrainLens Report\n\n- eval_loss=1.2",
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


def test_report_prompt_template_redacts_secrets_before_llm_prompt():
    prompt = ReportPromptTemplate().render(
        ReportPromptContext(
            markdown_report="## TrainLens Report\n\n- api_key=sk-test1234567890",
        )
    )

    assert "sk-test1234567890" not in prompt
    assert "[REDACTED]" in prompt
