from __future__ import annotations

import pytest

from trainlens import MultimodalDatasetExplanation, explain_dataset


class FakeImage:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        mode: str = "RGB",
        image_format: str = "PNG",
    ) -> None:
        self.size = (width, height)
        self.mode = mode
        self.format = image_format

    def getbands(self) -> tuple[str, ...]:
        return tuple(self.mode)


class FakeArray:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_explain_dataset_profiles_image_metadata_without_pixels() -> None:
    explanation = explain_dataset(
        {
            "image": [
                FakeImage(224, 224),
                FakeImage(256, 192),
                FakeImage(224, 224),
            ],
            "label": ["cat", "dog", "cat"],
        },
        target="label",
        name="train",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    split = explanation.splits[0]
    assert split.features[0].name == "image"
    assert split.features[0].kind == "image"
    assert explanation.modality_summaries[0].modalities == ("image",)
    image = explanation.modality_summaries[0].image_features[0]
    assert image.image_count == 3
    assert image.width_min == 224
    assert image.width_max == 256
    assert image.height_min == 192
    assert image.height_max == 224
    assert image.channels == (3,)
    assert image.has_variable_dimensions
    assert "Pixel values, image bytes, and file paths are not included" in explanation.markdown


def test_explain_dataset_detects_image_text_multimodal_split() -> None:
    explanation = explain_dataset(
        {
            "image": [FakeImage(128, 128), FakeImage(128, 128)],
            "caption": [
                "A detailed caption describing the first image for a vision-language task.",
                "A second detailed caption describing the paired visual example in the dataset.",
            ],
            "label": [0, 1],
        },
        target="label",
        name="train",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    modalities = explanation.modality_summaries[0]
    assert modalities.modalities == ("image", "text")
    assert modalities.is_multimodal
    assert explanation.is_multimodal
    assert any(
        "combines image data with text" in item
        for item in explanation.splits[0].observations
    )


def test_explain_dataset_supports_array_shaped_images_and_multiple_images_per_row() -> None:
    explanation = explain_dataset(
        {
            "images": [
                [FakeArray((3, 224, 224)), FakeArray((3, 224, 224))],
                [FakeArray((224, 224, 3))],
            ],
            "question": [
                "Compare the two images and explain the visual difference in detail.",
                "Describe the content of the single image and answer the question carefully.",
            ],
            "label": ["a", "b"],
        },
        target="label",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    image = explanation.modality_summaries[0].image_features[0]
    assert image.image_count == 3
    assert image.images_per_row_min == 1
    assert image.images_per_row_max == 2
    assert image.width_mean == pytest.approx(224.0)
    assert image.height_mean == pytest.approx(224.0)
    assert image.channels == (3,)
    assert "multiple images per row" in explanation.markdown


def test_explain_dataset_recognizes_hugging_face_image_references_without_leaking_paths() -> None:
    private_path = "/private/customer-42/images/example-001.jpg"
    explanation = explain_dataset(
        {
            "image": [
                {"path": private_path, "bytes": b"not-decoded"},
                {"path": "/private/customer-42/images/example-002.png", "bytes": b"not-decoded"},
            ],
            "label": [0, 1],
        },
        target="label",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    image = explanation.modality_summaries[0].image_features[0]
    assert image.metadata_count == 0
    assert image.formats == ("JPG", "PNG")
    assert private_path not in explanation.markdown
    assert "example-001.jpg" not in explanation.markdown
    assert "dimensions were not available" in explanation.markdown


def test_non_visual_dataset_preserves_original_dataset_explanation_behavior() -> None:
    explanation = explain_dataset(
        {
            "length": [10, 20, 30],
            "label": [0, 1, 1],
        },
        target="label",
    )

    assert not isinstance(explanation, MultimodalDatasetExplanation)
    assert explanation.splits[0].features[0].kind == "numeric"
    assert explanation.splits[0].features[0].mean == pytest.approx(20.0)
