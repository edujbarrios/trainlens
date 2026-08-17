from trainlens.llm.prompts import render_prompt_with_options
from trainlens.prompt_recipes import (
    ablation_study_prompt,
    controlled_experiment_prompt,
    hypothesis_test_prompt,
    improvement_plan_prompt,
    overfitting_review_prompt,
    prompt_options,
    reproducibility_audit_prompt,
    scientific_report_prompt,
    sensitivity_analysis_prompt,
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


def test_improvement_plan_recipe_renders_prioritized_actions():
    options = improvement_plan_prompt(model_family="multimodal projector")

    prompt = render_prompt_with_options("validation_loss=0.58", options=options)

    assert "Prioritize evidence-backed improvements" in prompt
    assert "multimodal projector" in prompt
    assert "expected information value and cost" in prompt
    assert "measurable success criterion" in prompt


def test_controlled_experiment_recipe_can_override_every_parameter():
    options = controlled_experiment_prompt(
        objective="Test whether a lower learning rate improves recall.",
        heading="## Recall Experiment",
        model_family="text classifier",
        audience="NLP researchers",
        tone="compact",
        rules=("Change learning rate only.",),
        focus_areas=("minority-class recall",),
        return_instructions=("Return a fixed seed and stopping rule.",),
    )

    assert options.objective == "Test whether a lower learning rate improves recall."
    assert options.heading == "## Recall Experiment"
    assert options.model_family == "text classifier"
    assert options.audience == "NLP researchers"
    assert options.tone == "compact"
    assert options.rules == ("Change learning rate only.",)
    assert options.focus_areas == ("minority-class recall",)
    assert options.return_instructions == ("Return a fixed seed and stopping rule.",)


def test_hypothesis_test_recipe_renders_scientific_decision_rules():
    prompt = render_prompt_with_options(
        "accuracy_mean=0.91\naccuracy_std=0.02",
        options=hypothesis_test_prompt(model_family="vision transformer"),
    )

    assert "null and alternative hypotheses" in prompt
    assert "vision transformer" in prompt
    assert "analysis method" in prompt
    assert "threats to validity" in prompt


def test_ablation_recipe_supports_complete_parameterization():
    options = ablation_study_prompt(
        objective="Measure the projector contribution.",
        heading="## Projector Ablation",
        model_family="vision-language model",
        audience="multimodal researchers",
        tone="compact and quantitative",
        rules=("Keep the encoder frozen.",),
        focus_areas=("projector depth",),
        return_instructions=("Return a three-row ablation matrix.",),
    )

    assert options.objective == "Measure the projector contribution."
    assert options.heading == "## Projector Ablation"
    assert options.model_family == "vision-language model"
    assert options.audience == "multimodal researchers"
    assert options.tone == "compact and quantitative"
    assert options.rules == ("Keep the encoder frozen.",)
    assert options.focus_areas == ("projector depth",)
    assert options.return_instructions == ("Return a three-row ablation matrix.",)


def test_sensitivity_recipe_defines_robustness_evidence():
    prompt = render_prompt_with_options(
        "learning_rate=0.001\nbatch_size=32",
        options=sensitivity_analysis_prompt(),
    )

    assert "parameter ranges and sampling scales" in prompt
    assert "robust operating regions" in prompt
    assert "sampling design" in prompt
    assert "summary statistics" in prompt


def test_reproducibility_recipe_supports_complete_parameterization():
    options = reproducibility_audit_prompt(
        objective="Audit the reported benchmark.",
        heading="## Benchmark Reproduction",
        model_family="language model",
        audience="external reviewers",
        tone="strict",
        rules=("Do not infer missing seeds.",),
        focus_areas=("environment capture",),
        return_instructions=("Return a replication checklist.",),
    )

    assert options.objective == "Audit the reported benchmark."
    assert options.heading == "## Benchmark Reproduction"
    assert options.model_family == "language model"
    assert options.audience == "external reviewers"
    assert options.tone == "strict"
    assert options.rules == ("Do not infer missing seeds.",)
    assert options.focus_areas == ("environment capture",)
    assert options.return_instructions == ("Return a replication checklist.",)
