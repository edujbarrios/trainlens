from trainlens.llm.prompts import render_prompt_with_options
from trainlens.prompt_recipes import (
    overfitting_review_prompt,
    prompt_options,
    scientific_report_prompt,
    training_diagnosis_prompt,
)


def test_scientific_report_recipe_can_override_every_parameter():
    options = scientific_report_prompt(
        objective="Explain the validation plateau.",
        heading="## Plateau Review",
        model_family="vision transformer",
        audience="computer-vision researchers",
        tone="concise and skeptical",
        rules=("Use supplied values only.",),
        focus_areas=("learning-rate schedule",),
        return_instructions=("Return three evidence-ranked hypotheses.",),
    )

    prompt = render_prompt_with_options("val_loss=0.42", options=options)

    assert "Explain the validation plateau." in prompt
    assert "## Plateau Review" in prompt
    assert "vision transformer" in prompt
    assert "computer-vision researchers" in prompt
    assert "concise and skeptical" in prompt
    assert "Use supplied values only." in prompt
    assert "learning-rate schedule" in prompt
    assert "three evidence-ranked hypotheses" in prompt


def test_generic_recipe_requires_and_preserves_all_prompt_fields():
    options = prompt_options(
        prompt_name="experiment_design",
        objective="Test one variable.",
        heading="## Controlled Test",
        model_family="classifier",
        audience="ML engineers",
        tone="direct",
        rules=("Hold the control fixed.",),
        focus_areas=("reproducibility",),
        return_instructions=("Return a stopping rule.",),
    )

    assert options.prompt_name == "experiment_design"
    assert options.objective == "Test one variable."
    assert options.heading == "## Controlled Test"
    assert options.model_family == "classifier"
    assert options.audience == "ML engineers"
    assert options.tone == "direct"
    assert options.rules == ("Hold the control fixed.",)
    assert options.focus_areas == ("reproducibility",)
    assert options.return_instructions == ("Return a stopping rule.",)


def test_training_diagnosis_recipe_renders_ranked_hypotheses():
    options = training_diagnosis_prompt(model_family="CLIP fine-tune")

    prompt = render_prompt_with_options("train_loss=0.2\nval_loss=0.7", options=options)

    assert "Rank the most likely causes" in prompt
    assert "CLIP fine-tune" in prompt
    assert "Rank hypotheses by confidence." in prompt
    assert "what evidence would falsify each hypothesis" in prompt


def test_overfitting_recipe_supports_complete_customization():
    options = overfitting_review_prompt(
        objective="Review the gap after epoch 10.",
        heading="## Generalization Audit",
        model_family="image classifier",
        audience="model reviewers",
        tone="brief",
        rules=("Compare matched epochs.",),
        focus_areas=("validation drift",),
        return_instructions=("Return one controlled test.",),
    )

    assert options.objective == "Review the gap after epoch 10."
    assert options.heading == "## Generalization Audit"
    assert options.model_family == "image classifier"
    assert options.audience == "model reviewers"
    assert options.tone == "brief"
    assert options.rules == ("Compare matched epochs.",)
    assert options.focus_areas == ("validation drift",)
    assert options.return_instructions == ("Return one controlled test.",)
