# Security Policy

Please report suspected vulnerabilities privately by opening a GitHub security advisory or emailing the maintainer.

TrainLens is notebook software and may inspect objects in the active Python namespace. It does not send data to external services unless optional LLM explanation is explicitly requested and configured by the user.

Before outbound LLM requests, TrainLens applies best-effort redaction to notebook context. Redaction is a defense-in-depth measure, not a substitute for keeping credentials out of the notebook namespace and using normal secret-management practices.

## Supported versions

TrainLens is pre-1.0. Security fixes will target the main branch until the first stable release.
