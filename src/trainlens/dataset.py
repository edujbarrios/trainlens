"""Deterministic, dependency-light dataset explanations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from math import isnan
from numbers import Real
from statistics import fmean
from typing import Literal, TypeAlias

DatasetFeatureKind: TypeAlias = Literal[
    "numeric",
    "categorical",
    "text",
    "boolean",
    "mixed",
    "object",
    "unknown",
]
DatasetTargetKind: TypeAlias = Literal[
    "categorical",
    "continuous",
    "text",
    "boolean",
    "mixed",
    "unknown",
]


@dataclass(frozen=True)
class DatasetClassBalance:
    """One target value and its observed frequency."""

    label: str
    count: int
    fraction: float


@dataclass(frozen=True)
class DatasetFeatureSummary:
    """Aggregate evidence for one dataset feature."""

    name: str
    kind: DatasetFeatureKind
    count: int
    missing: int
    unique: int | None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    average_length: float | None = None

    @property
    def missing_fraction(self) -> float:
        """Return the observed fraction of missing values."""

        return 0.0 if self.count == 0 else self.missing / self.count


@dataclass(frozen=True)
class DatasetTargetSummary:
    """Aggregate evidence for a supervised target column."""

    name: str
    kind: DatasetTargetKind
    count: int
    missing: int
    unique: int | None
    classes: tuple[DatasetClassBalance, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    @property
    def majority_fraction(self) -> float | None:
        """Return the largest observed class share for a discrete target."""

        if not self.classes:
            return None
        return max(item.fraction for item in self.classes)


@dataclass(frozen=True)
class DatasetSplitSummary:
    """Deterministic aggregate summary for one dataset split."""

    name: str
    row_count: int
    features: tuple[DatasetFeatureSummary, ...]
    target: DatasetTargetSummary | None = None
    observations: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetExplanation:
    """Reviewable explanation of one dataset or a mapping of dataset splits."""

    splits: tuple[DatasetSplitSummary, ...]
    observations: tuple[str, ...]
    markdown: str

    def _repr_markdown_(self) -> str:
        return self.markdown


def explain_dataset(
    dataset: object,
    *,
    target: str | None = None,
    name: str | None = None,
    max_features: int = 50,
    max_categories: int = 20,
) -> DatasetExplanation:
    """Explain dataset structure and target evidence without sending raw rows anywhere.

    The input may be a mapping of columns, a sequence of row mappings, a pandas-like
    object exposing ``columns``, a Hugging Face-like object exposing ``column_names``,
    or a mapping of split names to any of those tabular forms.
    """

    _validate_limit("max_features", max_features, minimum=1)
    _validate_limit("max_categories", max_categories, minimum=2)
    split_inputs = _split_inputs(dataset, name=name)
    split_summaries = tuple(
        _summarize_split(
            split_name,
            split_dataset,
            target=target,
            max_features=max_features,
            max_categories=max_categories,
        )
        for split_name, split_dataset in split_inputs
    )
    observations = _cross_split_observations(split_summaries)
    return DatasetExplanation(
        splits=split_summaries,
        observations=observations,
        markdown=_render_dataset_explanation(split_summaries, observations),
    )


def _split_inputs(dataset: object, *, name: str | None) -> tuple[tuple[str, object], ...]:
    if isinstance(dataset, Mapping) and dataset and _looks_like_split_mapping(dataset):
        output: list[tuple[str, object]] = []
        for split_name, split_dataset in dataset.items():
            if not isinstance(split_name, str) or not split_name.strip():
                raise ValueError("dataset split names must be non-empty strings")
            output.append((split_name.strip(), split_dataset))
        return tuple(output)
    return ((name.strip() if name and name.strip() else "dataset", dataset),)


def _looks_like_split_mapping(dataset: Mapping[object, object]) -> bool:
    values = tuple(dataset.values())
    if not values:
        return False
    return all(_looks_tabular(value) for value in values)


def _looks_tabular(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value) and all(_is_sequence_like(item) for item in value.values())
    if _row_sequence(value) is not None:
        return True
    return _column_names(value) is not None


def _summarize_split(
    name: str,
    dataset: object,
    *,
    target: str | None,
    max_features: int,
    max_categories: int,
) -> DatasetSplitSummary:
    columns = _tabular_columns(dataset)
    if not columns:
        return DatasetSplitSummary(
            name=name,
            row_count=0,
            features=(),
            observations=(f"Split `{name}` is empty.",),
        )

    lengths = {len(values) for _, values in columns}
    if len(lengths) != 1:
        raise ValueError("dataset columns must have the same number of rows")
    row_count = lengths.pop()
    column_map = dict(columns)
    if target is not None and target not in column_map:
        raise ValueError(f"target column {target!r} was not found in dataset split {name!r}")

    feature_items = [
        (column_name, values)
        for column_name, values in columns
        if column_name != target
    ]
    features = tuple(
        _summarize_feature(column_name, values)
        for column_name, values in feature_items[:max_features]
    )
    observations: list[str] = []
    omitted_features = len(feature_items) - len(features)
    if omitted_features > 0:
        observations.append(
            f"{omitted_features} additional features were omitted by the max_features limit."
        )
    for feature in features:
        if feature.missing_fraction >= 0.10:
            observations.append(
                f"Feature `{feature.name}` has {feature.missing_fraction:.1%} missing values."
            )
        if feature.unique == 1 and feature.count > feature.missing:
            observations.append(f"Feature `{feature.name}` is constant in this split.")

    target_summary = None
    if target is not None:
        target_summary = _summarize_target(
            target,
            column_map[target],
            max_categories=max_categories,
        )
        observations.extend(_target_observations(target_summary))

    return DatasetSplitSummary(
        name=name,
        row_count=row_count,
        features=features,
        target=target_summary,
        observations=tuple(observations),
    )


def _summarize_feature(name: str, values: tuple[object, ...]) -> DatasetFeatureSummary:
    nonmissing = tuple(value for value in values if not _is_missing(value))
    kind = _feature_kind(nonmissing)
    unique = _unique_count(nonmissing)
    numeric = _numeric_values(nonmissing) if kind == "numeric" else ()
    text_values = tuple(value for value in nonmissing if isinstance(value, str))
    return DatasetFeatureSummary(
        name=name,
        kind=kind,
        count=len(values),
        missing=len(values) - len(nonmissing),
        unique=unique,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        mean=fmean(numeric) if numeric else None,
        average_length=(fmean(len(value) for value in text_values) if text_values else None),
    )


def _summarize_target(
    name: str,
    values: tuple[object, ...],
    *,
    max_categories: int,
) -> DatasetTargetSummary:
    nonmissing = tuple(value for value in values if not _is_missing(value))
    feature_kind = _feature_kind(nonmissing)
    unique = _unique_count(nonmissing)
    numeric = _numeric_values(nonmissing) if feature_kind == "numeric" else ()
    continuous = (
        feature_kind == "numeric"
        and unique is not None
        and unique > max(max_categories, max(10, len(nonmissing) // 5))
    )
    if continuous:
        return DatasetTargetSummary(
            name=name,
            kind="continuous",
            count=len(values),
            missing=len(values) - len(nonmissing),
            unique=unique,
            minimum=min(numeric) if numeric else None,
            maximum=max(numeric) if numeric else None,
            mean=fmean(numeric) if numeric else None,
        )

    target_kind: DatasetTargetKind
    if feature_kind in ("numeric", "categorical"):
        target_kind = "categorical"
    elif feature_kind == "boolean":
        target_kind = "boolean"
    elif feature_kind == "text":
        target_kind = "text"
    elif feature_kind == "mixed":
        target_kind = "mixed"
    else:
        target_kind = "unknown"
    counts = Counter(_safe_label(value) for value in nonmissing)
    total = sum(counts.values())
    classes = tuple(
        DatasetClassBalance(label=label, count=count, fraction=count / total if total else 0.0)
        for label, count in counts.most_common(max_categories)
    )
    return DatasetTargetSummary(
        name=name,
        kind=target_kind,
        count=len(values),
        missing=len(values) - len(nonmissing),
        unique=unique,
        classes=classes,
        minimum=min(numeric) if numeric else None,
        maximum=max(numeric) if numeric else None,
        mean=fmean(numeric) if numeric else None,
    )


def _target_observations(target: DatasetTargetSummary) -> tuple[str, ...]:
    observations: list[str] = []
    missing_fraction = 0.0 if target.count == 0 else target.missing / target.count
    if missing_fraction > 0:
        observations.append(f"Target `{target.name}` has {missing_fraction:.1%} missing values.")
    if target.classes:
        majority = max(target.classes, key=lambda item: item.fraction)
        if majority.fraction >= 0.70 and len(target.classes) > 1:
            observations.append(
                f"Target `{target.name}` is imbalanced in this split: the largest observed "
                f"class (`{majority.label}`) accounts for {majority.fraction:.1%} of rows."
            )
        elif len(target.classes) > 1:
            observations.append(
                f"Target `{target.name}` has {target.unique or len(target.classes)} observed "
                f"values; the largest class share is {majority.fraction:.1%}."
            )
    return tuple(observations)


def _cross_split_observations(splits: tuple[DatasetSplitSummary, ...]) -> tuple[str, ...]:
    if len(splits) < 2:
        return ()
    observations: list[str] = []
    reference = splits[0]
    reference_features = {feature.name for feature in reference.features}
    for split in splits[1:]:
        feature_names = {feature.name for feature in split.features}
        if feature_names != reference_features:
            missing = sorted(reference_features - feature_names)
            extra = sorted(feature_names - reference_features)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(f"`{item}`" for item in missing))
            if extra:
                details.append("additional " + ", ".join(f"`{item}`" for item in extra))
            observations.append(
                f"Split `{split.name}` has a different feature schema than `{reference.name}`: "
                + "; ".join(details)
                + "."
            )
        shift = _target_distribution_shift(reference, split)
        if shift is not None:
            label, delta = shift
            observations.append(
                f"Target distribution differs between `{reference.name}` and `{split.name}`: "
                f"class `{label}` changes by {delta:.1%} of rows."
            )
    return tuple(observations)


def _target_distribution_shift(
    left: DatasetSplitSummary,
    right: DatasetSplitSummary,
) -> tuple[str, float] | None:
    if left.target is None or right.target is None:
        return None
    if not left.target.classes or not right.target.classes:
        return None
    left_rates = {item.label: item.fraction for item in left.target.classes}
    right_rates = {item.label: item.fraction for item in right.target.classes}
    labels = set(left_rates) | set(right_rates)
    if not labels:
        return None
    label = max(
        labels,
        key=lambda item: abs(left_rates.get(item, 0.0) - right_rates.get(item, 0.0)),
    )
    delta = abs(left_rates.get(label, 0.0) - right_rates.get(label, 0.0))
    if delta < 0.10:
        return None
    return label, delta


def _tabular_columns(dataset: object) -> tuple[tuple[str, tuple[object, ...]], ...]:
    if isinstance(dataset, Mapping):
        columns: list[tuple[str, tuple[object, ...]]] = []
        for name, values in dataset.items():
            if not isinstance(name, str) or not name:
                raise ValueError("dataset column names must be non-empty strings")
            columns.append((name, _sequence_values(values, column_name=name)))
        return tuple(columns)

    rows = _row_sequence(dataset)
    if rows is not None:
        row_names: list[str] = []
        for row in rows:
            for key in row:
                if not isinstance(key, str) or not key:
                    raise ValueError("dataset row keys must be non-empty strings")
                if key not in row_names:
                    row_names.append(key)
        return tuple((name, tuple(row.get(name) for row in rows)) for name in row_names)

    column_names = _column_names(dataset)
    if column_names is not None:
        columns = []
        for name in column_names:
            try:
                values = dataset[name]  # type: ignore[index]
            except (KeyError, TypeError, AttributeError) as exc:
                raise TypeError(f"could not read dataset column {name!r}") from exc
            columns.append((name, _sequence_values(values, column_name=name)))
        return tuple(columns)

    raise TypeError(
        "dataset must be a mapping of columns, a sequence of row mappings, "
        "or expose pandas-like `columns` / Hugging Face-like `column_names`"
    )


def _column_names(dataset: object) -> tuple[str, ...] | None:
    for attribute in ("column_names", "columns"):
        value = getattr(dataset, attribute, None)
        if value is None or isinstance(value, str | bytes):
            continue
        try:
            names = tuple(value)
        except TypeError:
            continue
        if names and all(isinstance(item, str) and item for item in names):
            return tuple(item for item in names if isinstance(item, str))
    return None


def _row_sequence(dataset: object) -> tuple[Mapping[object, object], ...] | None:
    if isinstance(dataset, str | bytes | bytearray | Mapping):
        return None
    if not isinstance(dataset, Sequence):
        return None
    rows = tuple(dataset)
    if not rows:
        return ()
    if all(isinstance(row, Mapping) for row in rows):
        return tuple(row for row in rows if isinstance(row, Mapping))
    return None


def _sequence_values(value: object, *, column_name: str) -> tuple[object, ...]:
    if not _is_sequence_like(value):
        raise TypeError(f"dataset column {column_name!r} must be a sequence of values")
    if isinstance(value, Sequence):
        return tuple(value)
    if isinstance(value, Iterable):
        return tuple(value)
    raise TypeError(f"dataset column {column_name!r} must be iterable")


def _is_sequence_like(value: object) -> bool:
    if isinstance(value, str | bytes | bytearray | Mapping):
        return False
    return isinstance(value, Iterable) and isinstance(value, Sized)


def _feature_kind(values: tuple[object, ...]) -> DatasetFeatureKind:
    if not values:
        return "unknown"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, Real) for value in values):
        return "numeric"
    if all(isinstance(value, str) for value in values):
        unique = _unique_count(values) or 0
        text_values = tuple(value for value in values if isinstance(value, str))
        average_length = fmean(len(value) for value in text_values)
        if average_length >= 32 or unique / len(values) > 0.50:
            return "text"
        return "categorical"
    types = {type(value) for value in values}
    return "object" if len(types) == 1 else "mixed"


def _numeric_values(values: tuple[object, ...]) -> tuple[float, ...]:
    output: list[float] = []
    for value in values:
        if isinstance(value, Real):
            numeric = float(value)
            if not isnan(numeric):
                output.append(numeric)
    return tuple(output)


def _unique_count(values: tuple[object, ...]) -> int | None:
    try:
        return len({_identity_token(value) for value in values})
    except (TypeError, ValueError):
        return None


def _identity_token(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _safe_label(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > 80:
        return text[:77] + "..."
    return text or "<empty>"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, Real):
        try:
            return isnan(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    return False


def _render_dataset_explanation(
    splits: tuple[DatasetSplitSummary, ...],
    observations: tuple[str, ...],
) -> str:
    lines = [
        "# TrainLens Dataset Explanation",
        "",
        "This is deterministic aggregate evidence about the dataset. It does not prove "
        "that any dataset property caused a model result.",
        "",
    ]
    for split in splits:
        lines.extend([f"## Split: {split.name}", "", f"- rows: {split.row_count}"])
        lines.append(f"- summarized features: {len(split.features)}")
        if split.features:
            lines.extend(["", "### Features"])
            for feature in split.features:
                details = [
                    f"kind={feature.kind}",
                    f"missing={feature.missing}/{feature.count}",
                ]
                if feature.unique is not None:
                    details.append(f"unique={feature.unique}")
                if feature.minimum is not None and feature.maximum is not None:
                    details.append(f"range={feature.minimum:.6g}..{feature.maximum:.6g}")
                if feature.mean is not None:
                    details.append(f"mean={feature.mean:.6g}")
                if feature.average_length is not None:
                    details.append(f"avg_length={feature.average_length:.1f}")
                lines.append(f"- `{feature.name}`: " + ", ".join(details))
        if split.target is not None:
            target = split.target
            lines.extend(["", "### Target"])
            details = [
                f"kind={target.kind}",
                f"missing={target.missing}/{target.count}",
            ]
            if target.unique is not None:
                details.append(f"unique={target.unique}")
            lines.append(f"- `{target.name}`: " + ", ".join(details))
            if target.minimum is not None and target.maximum is not None:
                lines.append(
                    f"- numeric range: {target.minimum:.6g}..{target.maximum:.6g}, "
                    f"mean={target.mean:.6g}"
                )
            if target.classes:
                lines.append("- observed class distribution:")
                lines.extend(
                    f"  - `{item.label}`: {item.count} ({item.fraction:.1%})"
                    for item in target.classes
                )
        if split.observations:
            lines.extend(["", "### Observations"])
            lines.extend(f"- {item}" for item in split.observations)
        lines.append("")
    if observations:
        lines.extend(["## Cross-split observations", ""])
        lines.extend(f"- {item}" for item in observations)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _validate_limit(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
