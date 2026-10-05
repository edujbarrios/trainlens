"""Small dependency-free example of TrainLens multimodal dataset evidence."""

from __future__ import annotations

from trainlens import explain_dataset


class ImageMetadata:
    """Stand-in for a PIL-like image object already present in a notebook."""

    def __init__(self, width: int, height: int, image_format: str = "JPEG") -> None:
        self.size = (width, height)
        self.mode = "RGB"
        self.format = image_format


dataset_context = explain_dataset(
    {
        "train": {
            "image": [ImageMetadata(224, 224), ImageMetadata(256, 192)],
            "caption": [
                "A long caption describing the first visual training example in detail.",
                "A long caption describing the second visual training example in detail.",
            ],
            "label": [0, 1],
        },
        "validation": {
            "image": [ImageMetadata(256, 256), ImageMetadata(256, 256)],
            "caption": [
                "A long validation caption paired with the first image in this split.",
                "A long validation caption paired with the second image in this split.",
            ],
            "label": [0, 1],
        },
    },
    target="label",
)

print(dataset_context.markdown)
