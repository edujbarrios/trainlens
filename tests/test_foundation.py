from trainlens.analyzers.metrics import extract_metric_series
from trainlens.heuristics.foundation import (
    detect_adapter_pressure,
    detect_foundation_architecture,
    detect_loss_plateau,
)
from trainlens.pipeline import explain_namespace


class LlavaLikeModel:
    vision_tower = object()
    language_model = object()
    mm_projector = object()

    class Config:
        model_type = "llava"

    config = Config()


class MusicGenerationModel:
    text_encoder = object()
    audio_autoencoder = object()
    diffusion_unet = object()

    class Config:
        model_type = "musicgen"

    config = Config()


def test_detects_vlm_projector_profile():
    families = detect_foundation_architecture(LlavaLikeModel(), {"mm_projector": object()})

    assert "llm" in families
    assert "projector" in families
    assert "vlm" in families


def test_normalizes_eval_loss_alias_for_fine_tuning():
    series = extract_metric_series({"trainer_history": {"eval_loss": [1.8, 1.7]}})

    assert series["validation_loss"].last == 1.7


def test_detects_loss_plateau_and_small_adapter_rank():
    series = extract_metric_series({"history": {"eval_loss": [1.8, 1.72, 1.71, 1.705]}})

    assert detect_loss_plateau(series["validation_loss"]) is not None
    assert detect_adapter_pressure({"lora_rank": 4}) is not None


def test_pipeline_recommends_vlm_validation_workflow():
    result = explain_namespace(
        {
            "model": LlavaLikeModel(),
            "history": {"train_loss": [2.1, 1.7, 1.6, 1.6], "eval_loss": [2.0, 1.8, 1.79, 1.79]},
            "lora_rank": 4,
        }
    )

    assert any("Foundation-model profile" in item for item in result.summary)
    assert any("projector alignment" in item.action.lower() for item in result.recommendations)


def test_pipeline_recommends_music_generation_metrics():
    result = explain_namespace(
        {
            "model": MusicGenerationModel(),
            "history": {
                "train_loss": [1.8, 1.4, 1.2, 1.19],
                "eval_loss": [1.7, 1.5, 1.45, 1.44],
                "eval_clap_score": [0.22, 0.25, 0.29, 0.31],
            },
            "sample_rate": 44_100,
        }
    )

    assert any("MUSIC" in item for item in result.summary)
    assert any("FAD" in item.action for item in result.recommendations)
