"""Image-aware extensions for deterministic dataset explanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePath
from statistics import fmean
from typing import TypeGuard, cast

from trainlens.dataset import (
    DatasetExplanation,
    DatasetFeatureKind,
    DatasetFeatureSummary,
    DatasetSplitSummary,
    _cross_split_observations,
    _is_missing,
    _render_dataset_explanation,
    _split_inputs,
    _summarize_feature,
    _summarize_target,
    _tabular_columns,
    _target_observations,
    _validate_limit,
)
from trainlens.dataset import explain_dataset as _explain_tabular_dataset

_IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class DatasetImageSummary:
    """Aggregate, pixel-free evidence for one image feature."""

    feature_name: str
    row_count: int
    rows_with_images: int
    image_count: int
    metadata_count: int
    images_per_row_min: int
    images_per_row_max: int
    images_per_row_mean: float
    width_min: int | None = None
    width_max: int | None = None
    width_mean: float | None = None
    height_min: int | None = None
    height_max: int | None = None
    height_mean: float | None = None
    aspect_ratio_min: float | None = None
    aspect_ratio_max: float | None = None
    aspect_ratio_mean: float | None = None
    channels: tuple[int, ...] = ()
    modes: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()

    @property
    def has_variable_dimensions(self) -> bool:
        """Return whether inspected images use more than one width or height."""

        return (
            self.width_min is not None
            and self.width_max is not None
            and self.height_min is not None
            and self.height_max is not None
            and (self.width_min != self.width_max or self.height_min != self.height_max)
        )


@dataclass(frozen=True)
class DatasetModalitySummary:
    """Aggregate modality evidence for one split."""

    split_name: str
    modalities: tuple[str, ...]
    image_features: tuple[DatasetImageSummary, ...] = ()

    @property
    def is_multimodal(self) -> bool:
        """Return whether visual data is combined with another input modality."""

        return "image" in self.modalities and len(self.modalities) > 1


@dataclass(frozen=True)
class MultimodalDatasetExplanation(DatasetExplanation):
    """Dataset explanation enriched with aggregate image and modality evidence."""

    modality_summaries: tuple[DatasetModalitySummary, ...] = ()

    @property
    def is_multimodal(self) -> bool:
        """Return whether at least one split combines images with another modality."""

        return any(summary.is_multimodal for summary in self.modality_summaries)


@dataclass(frozen=True)
class _ImageMetadata:
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    mode: str | None = None
    format: str | None = None


def explain_dataset(
    dataset: object,
    *,
    target: str | None = None,
    name: str | None = None,
    max_features: int = 50,
    max_categories: int = 20,
) -> DatasetExplanation:
    """Explain tabular, image, and image-centered multimodal datasets locally.

    Image features are detected through dependency-light metadata inspection. Supported
    representations include PIL-like objects, NumPy/PyTorch-like arrays exposing ``shape``,
    common image-path strings, Hugging Face-style ``{"path", "bytes"}`` image values, and
    rows containing multiple image objects. Pixel values, image bytes, and file paths are
    never rendered into the explanation.
    """

    _validate_limit("max_features", max_features, minimum=1)
    _validate_limit("max_categories", max_categories, minimum=2)
    split_inputs = _split_inputs(dataset, name=name)

    split_summaries: list[DatasetSplitSummary] = []
    modality_summaries: list[DatasetModalitySummary] = []
    for split_name, split_dataset in split_inputs:
        split, modalities = _summarize_split(
            split_name,
            split_dataset,
            target=target,
            max_features=max_features,
            max_categories=max_categories,
        )
        split_summaries.append(split)
        modality_summaries.append(modalities)

    if not any(summary.image_features for summary in modality_summaries):
        return _explain_tabular_dataset(
            dataset,
            target=target,
            name=name,
            max_features=max_features,
            max_categories=max_categories,
        )

    splits = tuple(split_summaries)
    modality_tuple = tuple(modality_summaries)
    observations = _cross_split_observations(splits) + _visual_cross_split_observations(
        modality_tuple
    )
    base_markdown = _render_dataset_explanation(splits, observations)
    markdown = base_markdown.rstrip() + "\n\n" + _render_modality_evidence(modality_tuple)
    return MultimodalDatasetExplanation(
        splits=splits,
        observations=observations,
        markdown=markdown.rstrip() + "\n",
        modality_summaries=modality_tuple,
    )


def _summarize_split(
    name: str,
    dataset: object,
    *,
    target: str | None,
    max_features: int,
    max_categories: int,
) -> tuple[DatasetSplitSummary, DatasetModalitySummary]:
    columns = _tabular_columns(dataset)
    if not columns:
        split = DatasetSplitSummary(
            name=name,
            row_count=0,
            features=(),
            observations=(f"Split `{name}` is empty.",),
        )
        return split, DatasetModalitySummary(split_name=name, modalities=())

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
    feature_summaries: list[DatasetFeatureSummary] = []
    image_summaries: list[DatasetImageSummary] = []
    for column_name, values in feature_items[:max_features]:
        image_summary = _summarize_image_feature(column_name, values)
        if image_summary is None:
            feature_summaries.append(_summarize_feature(column_name, values))
            continue
        feature_summaries.append(
            DatasetFeatureSummary(
                name=column_name,
                kind=cast(DatasetFeatureKind, "image"),
                count=len(values),
                missing=len(values) - image_summary.rows_with_images,
                unique=None,
            )
        )
        image_summaries.append(image_summary)

    features = tuple(feature_summaries)
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
    for image in image_summaries:
        observations.extend(_image_observations(image))

    target_summary = None
    if target is not None:
        target_summary = _summarize_target(
            target,
            column_map[target],
            max_categories=max_categories,
        )
        observations.extend(_target_observations(target_summary))

    modalities = _feature_modalities(features)
    modality_summary = DatasetModalitySummary(
        split_name=name,
        modalities=modalities,
        image_features=tuple(image_summaries),
    )
    if modality_summary.is_multimodal:
        other_modalities = " and ".join(item for item in modalities if item != "image")
        observations.append(
            f"Split `{name}` combines image data with {other_modalities} features."
        )

    split = DatasetSplitSummary(
        name=name,
        row_count=row_count,
        features=features,
        target=target_summary,
        observations=tuple(observations),
    )
    return split, modality_summary


def _summarize_image_feature(
    name: str,
    values: tuple[object, ...],
) -> DatasetImageSummary | None:
    nonmissing = tuple(value for value in values if not _is_missing(value))
    if not nonmissing:
        return None

    rows: list[tuple[_ImageMetadata, ...]] = []
    for value in nonmissing:
        metadata = _image_items(value)
        if not metadata:
            return None
        rows.append(metadata)

    images = tuple(item for row in rows for item in row)
    widths = tuple(item.width for item in images if item.width is not None)
    heights = tuple(item.height for item in images if item.height is not None)
    ratios = tuple(
        item.width / item.height
        for item in images
        if item.width is not None and item.height is not None and item.height > 0
    )
    channels = tuple(sorted({item.channels for item in images if item.channels is not None}))
    modes = tuple(sorted({item.mode for item in images if item.mode is not None}))
    formats = tuple(sorted({item.format for item in images if item.format is not None}))
    images_per_row = tuple(len(row) for row in rows)
    metadata_count = sum(
        1 for item in images if item.width is not None and item.height is not None
    )
    return DatasetImageSummary(
        feature_name=name,
        row_count=len(values),
        rows_with_images=len(rows),
        image_count=len(images),
        metadata_count=metadata_count,
        images_per_row_min=min(images_per_row),
        images_per_row_max=max(images_per_row),
        images_per_row_mean=fmean(images_per_row),
        width_min=min(widths) if widths else None,
        width_max=max(widths) if widths else None,
        width_mean=fmean(widths) if widths else None,
        height_min=min(heights) if heights else None,
        height_max=max(heights) if heights else None,
        height_mean=fmean(heights) if heights else None,
        aspect_ratio_min=min(ratios) if ratios else None,
        aspect_ratio_max=max(ratios) if ratios else None,
        aspect_ratio_mean=fmean(ratios) if ratios else None,
        channels=channels,
        modes=modes,
        formats=formats,
    )


def _image_items(value: object) -> tuple[_ImageMetadata, ...]:
    metadata = _image_metadata(value)
    if metadata is not None:
        return (metadata,)
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray | Mapping
    ):
        if not value:
            return ()
        output: list[_ImageMetadata] = []
        for item in value:
            item_metadata = _image_metadata(item)
            if item_metadata is None:
                return ()
            output.append(item_metadata)
        return tuple(output)
    return ()


def _image_metadata(value: object) -> _ImageMetadata | None:
    if isinstance(value, str):
        image_format = _format_from_path(value)
        if image_format is not None:
            return _ImageMetadata(format=image_format)
        return None

    if isinstance(value, Mapping):
        path = value.get("path")
        raw_bytes = value.get("bytes")
        array = value.get("array")
        if array is not None:
            shape_metadata = _metadata_from_shape(array)
            if shape_metadata is not None:
                return shape_metadata
        if isinstance(path, str) and (
            _format_from_path(path) is not None or raw_bytes is not None
        ):
            return _ImageMetadata(format=_format_from_path(path))
        if raw_bytes is not None and "path" in value:
            return _ImageMetadata()
        return None

    pil_metadata = _metadata_from_pil_like(value)
    if pil_metadata is not None:
        return pil_metadata
    return _metadata_from_shape(value)


def _metadata_from_pil_like(value: object) -> _ImageMetadata | None:
    size = getattr(value, "size", None)
    if not isinstance(size, tuple) or len(size) != 2:
        return None
    width, height = size
    if not _positive_int(width) or not _positive_int(height):
        return None

    mode_value = getattr(value, "mode", None)
    mode = mode_value if isinstance(mode_value, str) and mode_value else None
    format_value = getattr(value, "format", None)
    image_format = format_value if isinstance(format_value, str) and format_value else None
    channels = _channels_from_mode(mode)
    if channels is None:
        getbands = getattr(value, "getbands", None)
        if callable(getbands):
            try:
                bands = tuple(getbands())
            except (TypeError, ValueError, AttributeError):
                bands = ()
            if bands:
                channels = len(bands)
    return _ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        mode=mode,
        format=image_format,
    )


def _metadata_from_shape(value: object) -> _ImageMetadata | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dimensions = tuple(int(item) for item in shape)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(dimensions) == 2:
        height, width = dimensions
        if min(height, width) < 8:
            return None
        return _ImageMetadata(width=width, height=height, channels=1, mode="grayscale")
    if len(dimensions) != 3:
        return None

    first, middle, last = dimensions
    if last in (1, 2, 3, 4) and first >= 8 and middle >= 8:
        return _ImageMetadata(width=middle, height=first, channels=last)
    if first in (1, 2, 3, 4) and middle >= 8 and last >= 8:
        return _ImageMetadata(width=last, height=middle, channels=first)
    return None


def _format_from_path(path: str) -> str | None:
    suffix = PurePath(path).suffix.lower()
    if suffix not in _IMAGE_EXTENSIONS:
        return None
    return suffix.lstrip(".").upper()


def _channels_from_mode(mode: str | None) -> int | None:
    if mode is None:
        return None
    normalized = mode.upper()
    if normalized in {"1", "L", "I", "F", "P"}:
        return 1
    if normalized in {"RGB", "YCBCR", "LAB", "HSV"}:
        return 3
    if normalized in {"RGBA", "CMYK"}:
        return 4
    return None


def _positive_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _feature_modalities(features: tuple[DatasetFeatureSummary, ...]) -> tuple[str, ...]:
    modalities: list[str] = []
    for feature in features:
        modality: str | None
        if str(feature.kind) == "image":
            modality = "image"
        elif feature.kind == "text":
            modality = "text"
        elif feature.kind != "unknown":
            modality = "tabular"
        else:
            modality = None
        if modality is not None and modality not in modalities:
            modalities.append(modality)
    preferred_order = ("image", "text", "tabular")
    return tuple(item for item in preferred_order if item in modalities)


def _image_observations(image: DatasetImageSummary) -> tuple[str, ...]:
    observations: list[str] = []
    if image.metadata_count == 0:
        observations.append(
            f"Image feature `{image.feature_name}` was recognized, but dimensions were not "
            "available without decoding the underlying media."
        )
    elif image.has_variable_dimensions:
        observations.append(
            f"Image feature `{image.feature_name}` has variable dimensions: widths span "
            f"{image.width_min}..{image.width_max}px and heights span "
            f"{image.height_min}..{image.height_max}px."
        )
    if image.images_per_row_max > 1:
        observations.append(
            f"Image feature `{image.feature_name}` contains multiple images per row "
            f"(up to {image.images_per_row_max})."
        )
    if len(image.formats) > 1:
        observations.append(
            f"Image feature `{image.feature_name}` uses multiple observed formats: "
            + ", ".join(image.formats)
            + "."
        )
    return tuple(observations)


def _visual_cross_split_observations(
    summaries: tuple[DatasetModalitySummary, ...],
) -> tuple[str, ...]:
    if len(summaries) < 2:
        return ()
    observations: list[str] = []
    reference = summaries[0]
    reference_images = {item.feature_name: item for item in reference.image_features}
    for summary in summaries[1:]:
        image_map = {item.feature_name: item for item in summary.image_features}
        if set(image_map) != set(reference_images):
            observations.append(
                f"Visual feature schema differs between `{reference.split_name}` and "
                f"`{summary.split_name}`."
            )
        for feature_name in sorted(set(reference_images) & set(image_map)):
            left = reference_images[feature_name]
            right = image_map[feature_name]
            change = _dimension_change(left, right)
            if change is not None and change >= 0.25:
                observations.append(
                    f"Image dimensions differ materially for `{feature_name}` between "
                    f"`{reference.split_name}` and `{summary.split_name}` "
                    f"({change:.0%} relative mean-size change)."
                )
    return tuple(observations)


def _dimension_change(left: DatasetImageSummary, right: DatasetImageSummary) -> float | None:
    if (
        left.width_mean is None
        or left.height_mean is None
        or right.width_mean is None
        or right.height_mean is None
    ):
        return None
    left_area = left.width_mean * left.height_mean
    right_area = right.width_mean * right.height_mean
    if left_area <= 0:
        return None
    return abs(right_area - left_area) / left_area


def _render_modality_evidence(summaries: tuple[DatasetModalitySummary, ...]) -> str:
    lines = [
        "## Modalities and visual evidence",
        "",
        "Only aggregate image metadata is shown. Pixel values, image bytes, and file paths are "
        "not included.",
    ]
    for summary in summaries:
        if not summary.image_features:
            continue
        lines.extend(["", f"### Split: {summary.split_name}"])
        lines.append("- modalities: " + " + ".join(summary.modalities))
        multimodal = "yes" if summary.is_multimodal else "no"
        lines.append(f"- image-centered multimodal: {multimodal}")
        for image in summary.image_features:
            details = [
                f"rows_with_images={image.rows_with_images}/{image.row_count}",
                f"images={image.image_count}",
                f"images_per_row={image.images_per_row_min}..{image.images_per_row_max}",
            ]
            if image.width_min is not None and image.width_max is not None:
                details.append(f"width={image.width_min}..{image.width_max}px")
            if image.height_min is not None and image.height_max is not None:
                details.append(f"height={image.height_min}..{image.height_max}px")
            if image.aspect_ratio_mean is not None:
                details.append(f"mean_aspect_ratio={image.aspect_ratio_mean:.3g}")
            if image.channels:
                details.append("channels=" + ",".join(str(item) for item in image.channels))
            if image.modes:
                details.append("modes=" + ",".join(image.modes))
            if image.formats:
                details.append("formats=" + ",".join(image.formats))
            lines.append(f"- `{image.feature_name}`: " + ", ".join(details))
    return "\n".join(lines).strip() + "\n"
