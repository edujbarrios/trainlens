from trainlens.security import REDACTED_VALUE, redact_text, sanitize_value


def test_redact_text_removes_common_secret_shapes():
    text = (
        "Use api_key=sk-test1234567890 and "
        "Authorization: Bearer very-secret-token-value"
    )

    redacted = redact_text(text)

    assert "sk-test1234567890" not in redacted
    assert "very-secret-token-value" not in redacted
    assert redacted.count(REDACTED_VALUE) == 2


def test_sanitize_value_redacts_sensitive_nested_names():
    value = {
        "dataset": "mnist",
        "token": "plain-text-token",
        "nested": {"client_secret": "super-secret"},
    }

    sanitized = sanitize_value("config", value)

    assert sanitized["dataset"] == "mnist"
    assert sanitized["token"] == REDACTED_VALUE
    assert sanitized["nested"]["client_secret"] == REDACTED_VALUE
