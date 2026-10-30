from pathlib import Path

import pytest

from trainlens import load_prompt, prompt_text


def test_prompt_text_dedents_and_strips_multiline_prompt():
    text = prompt_text(
        """
        Explain what happened during this fine-tuning run.

        Prefer:
          - controlled experiments
          - low-cost changes first
        """
    )

    assert text == (
        "Explain what happened during this fine-tuning run.\n\n"
        "Prefer:\n"
        "  - controlled experiments\n"
        "  - low-cost changes first"
    )


def test_prompt_text_strips_outer_whitespace_from_single_line_text():
    assert prompt_text("  Keep the held-out test split untouched.  ") == (
        "Keep the held-out test split untouched."
    )


def test_load_prompt_reads_markdown_and_preserves_structure(tmp_path: Path):
    prompt_path = tmp_path / "analysis.md"
    prompt_path.write_text(
        "\n# Analysis objective\n\n- Compare validation and test.\n- Keep test held out.\n\n"
        "    preserve this indentation\n\n",
        encoding="utf-8",
    )

    assert load_prompt(prompt_path) == (
        "# Analysis objective\n\n"
        "- Compare validation and test.\n"
        "- Keep test held out.\n\n"
        "    preserve this indentation"
    )


def test_load_prompt_accepts_string_paths_and_case_insensitive_md_suffix(tmp_path: Path):
    prompt_path = tmp_path / "OBJECTIVE.MD"
    prompt_path.write_text("  Explain the run.  ", encoding="utf-8")

    assert load_prompt(str(prompt_path)) == "Explain the run."


def test_load_prompt_rejects_non_markdown_files(tmp_path: Path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Explain the run.", encoding="utf-8")

    with pytest.raises(ValueError, match=r"expects a \.md file"):
        load_prompt(prompt_path)


def test_load_prompt_substitutes_named_variables(tmp_path: Path):
    prompt_path = tmp_path / "plan.md"
    prompt_path.write_text(
        "Analyze $model on $dataset. Keep $$100 as a literal cost.\n",
        encoding="utf-8",
    )

    assert load_prompt(
        prompt_path, variables={"model": "Llama", "dataset": "validation"}
    ) == "Analyze Llama on validation. Keep $100 as a literal cost."


def test_load_prompt_without_variables_preserves_dollar_placeholders(tmp_path: Path):
    prompt_path = tmp_path / "plan.md"
    prompt_path.write_text("Analyze $model and charge $$100.", encoding="utf-8")

    assert load_prompt(prompt_path) == "Analyze $model and charge $$100."


def test_load_prompt_missing_variable_raises_key_error(tmp_path: Path):
    prompt_path = tmp_path / "plan.md"
    prompt_path.write_text("Analyze $model and $dataset.", encoding="utf-8")

    with pytest.raises(KeyError, match="dataset"):
        load_prompt(prompt_path, variables={"model": "Llama"})


def test_load_prompt_rejects_invalid_template_placeholder(tmp_path: Path):
    prompt_path = tmp_path / "plan.md"
    prompt_path.write_text("Invalid: $!.", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid placeholder"):
        load_prompt(prompt_path, variables={})
