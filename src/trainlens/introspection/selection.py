"""Helpers for selecting framework artifacts that belong to a chosen model."""

from __future__ import annotations

from typing import cast

from trainlens.models.snapshot import NotebookSnapshot


def framework_source_for_model(
    snapshot: NotebookSnapshot,
    framework: str,
    model_ref: object | None,
) -> object | None:
    """Return a framework source associated with ``model_ref`` when possible.

    When no model has been selected, the first matching framework source remains a
    sensible fallback. Once a concrete model is selected, however, returning an
    unrelated trainer can mix configuration from different experiments, so only an
    artifact that points at the same model object is accepted.
    """

    fallback: object | None = None
    for artifact in snapshot.framework_artifacts:
        if artifact.framework != framework:
            continue
        source = snapshot.raw_namespace.get(artifact.variable_name)
        if source is None:
            continue
        source = cast(object, source)
        if model_ref is None:
            if fallback is None:
                fallback = source
            continue
        if artifact.model_ref is model_ref:
            return source
    return fallback
