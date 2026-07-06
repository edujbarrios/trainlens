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


def test_redact_text_removes_url_credentials_and_query_secrets():
    text = (
        "Fetch https://user:pass@example.com/data?token=plain-token-value "
        "and https://example.com/callback?api_key=sk-test1234567890&ok=1"
    )

    redacted = redact_text(text)

    assert "user:pass" not in redacted
    assert "plain-token-value" not in redacted
    assert "sk-test1234567890" not in redacted
    assert "https://[REDACTED]@example.com/data?token=[REDACTED]" in redacted
    assert "api_key=[REDACTED]" in redacted


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
