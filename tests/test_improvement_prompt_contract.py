from trainlens import improvement_plan_prompt
from trainlens.llm.prompts import render_prompt_with_options


def test_improvement_recipe_requires_actionable_grounded_output() -> None:
    prompt = render_prompt_with_options(
        "validation_loss=0.72",
        options=improvement_plan_prompt(),
    )

    assert "Do not guess missing hyperparameter values" in prompt
    assert "evidence, proposed change, rationale, expected effect" in prompt
    assert "risk or cost, confidence" in prompt
    assert "measurable success criterion" in prompt
    assert "one recommended next run" in prompt
