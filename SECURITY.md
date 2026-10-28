# Security Policy

## Report a vulnerability

Please report security issues **privately** through the repository's [Security tab](https://github.com/edujbarrios/trainlens/security) (use private vulnerability reporting if available). Avoid opening a public issue containing exploit details, tokens, or private training data.

## Data handling

TrainLens inspects objects and metrics available in the active Python namespace. Local deterministic analysis and TrainLens Agent Mode do not themselves call an LLM service. Agent Mode produces context that a **separate agent** may process or transmit according to that agent's own settings.

Direct LLM reports make a request to the endpoint chosen by the user. TrainLens applies best-effort redaction and context limits before outbound requests, but these are **not** guarantees against disclosure. Review the request preview, keep credentials out of notebook state, and follow the provider's privacy policy.

## Supported versions

TrainLens is pre-1.0. Security fixes currently target the main branch rather than separately maintained older release branches.
