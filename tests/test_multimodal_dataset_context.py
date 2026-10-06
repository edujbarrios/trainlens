from __future__ import annotations

from trainlens import explain_dataset, preview_notebook_context
from trainlens.llm.verified import evidence_catalog
from trainlens.models.analysis import AnalysisResult


class FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.size = (width, height)
        self.mode = "RGB"
        self.format = "JPEG"


def test_multimodal_context_includes_visual_metadata_without_media_content() -> None:
    private_path = "/private/project-x/images/customer-frame-001.jpg"
    explanation = explain_dataset(
        {
            "image": [
                {"path": private_path, "bytes": b"private-image-bytes"},
                {"path": "/private/project-x/images/customer-frame-002.jpg", "bytes": b"more"},
            ],
            "caption": [
                "A sufficiently long caption describing the first multimodal example.",
                "A sufficiently long caption describing the second multimodal example.",
            ],
            "label": [0, 1],
        },
        target="label",
        name="train",
    )

    context = preview_notebook_context(
        {"history": {"loss": [1.0, 0.8], "val_loss": [1.1, 0.9]}},
        dataset_explanation=explanation,
    )

    assert "## Dataset Context" in context.markdown
    assert "## Modalities and visual evidence" in context.markdown
    assert "image + text" in context.markdown
    assert "Pixel values, image bytes, and file paths are not included" in context.markdown
    assert private_path not in context.markdown
    assert "customer-frame-001.jpg" not in context.markdown
    assert "private-image-bytes" not in context.markdown


def test_visual_feature_and_observations_are_available_as_verified_evidence() -> None:
    explanation = explain_dataset(
        {
            "image": [FakeImage(128, 128), FakeImage(256, 192)],
            "caption": [
                "A detailed text description paired with the first visual training example.",
                "A detailed text description paired with the second visual training example.",
            ],
            "label": [0, 1],
        },
        target="label",
        name="train",
    )

    catalog = evidence_catalog(AnalysisResult(), dataset_explanation=explanation)
    details = {item.evidence_id: item.detail for item in catalog}

    image_details = [
        detail for evidence_id, detail in details.items() if ":feature:image-" in evidence_id
    ]
    assert image_details
    assert "kind=image" in image_details[0]
    assert any("variable dimensions" in detail for detail in details.values())
    assert any("combines image data with text" in detail for detail in details.values())


def test_cross_split_visual_shift_becomes_stable_dataset_evidence() -> None:
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

    catalog = evidence_catalog(AnalysisResult(), dataset_explanation=explanation)
    cross_split = [
        item
        for item in catalog
        if item.evidence_id.startswith("dataset:observation:")
    ]

    assert cross_split
    assert any("Image dimensions differ materially" in item.detail for item in cross_split)
