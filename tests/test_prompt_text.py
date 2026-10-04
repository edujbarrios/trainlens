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
