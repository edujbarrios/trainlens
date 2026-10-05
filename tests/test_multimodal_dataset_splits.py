from __future__ import annotations

from trainlens import MultimodalDatasetExplanation, explain_dataset


class FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.mode = "RGB"
        self.format = "JPEG"


def test_image_tabular_dataset_is_marked_multimodal() -> None:
    explanation = explain_dataset(
        {
            "image": [FakeImage(128, 128), FakeImage(128, 128)],
            "age": [21, 37],
            "label": [0, 1],
        },
        target="label",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    summary = explanation.modality_summaries[0]
    assert summary.modalities == ("image", "tabular")
    assert summary.is_multimodal


def test_cross_split_image_resolution_difference_is_reported() -> None:
    explanation = explain_dataset(
        {
            "train": {
                "image": [FakeImage(128, 128), FakeImage(128, 128)],
                "label": [0, 1],
            },
            "validation": {
                "image": [FakeImage(256, 256), FakeImage(256, 256)],
                "label": [0, 1],
            },
        },
        target="label",
    )

    assert isinstance(explanation, MultimodalDatasetExplanation)
    assert any("Image dimensions differ materially" in item for item in explanation.observations)
