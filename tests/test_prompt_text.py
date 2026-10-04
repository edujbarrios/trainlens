from trainlens import prompt_text


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
