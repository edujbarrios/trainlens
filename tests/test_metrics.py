from trainlens.analyzers.metrics import extract_metric_series, paired_metric


def test_extracts_keras_style_history():
    series = extract_metric_series(
        {"history": {"accuracy": [0.7, 0.8], "val_accuracy": [0.68, 0.75]}}
    )

    train, validation = paired_metric(series, "accuracy")

    assert train is not None
    assert validation is not None
    assert train.last == 0.8
    assert validation.split == "validation"


def test_extracts_trainer_style_log_history():
    series = extract_metric_series(
        {
            "log_history": [
                {"step": 0, "loss": 2.4},
                {"step": 1, "loss": 2.1, "eval_loss": 2.3},
                {"step": 2, "loss": 1.9, "eval_loss": 2.0},
            ]
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.values == (2.4, 2.1, 1.9)
    assert train.steps == (0, 1, 2)
    assert validation.values == (2.3, 2.0)
    assert validation.steps == (1, 2)


def test_extracts_pytorch_loop_loss_lists():
    series = extract_metric_series(
        {
            "train_losses": [2.3, 1.8, 1.4],
            "val_losses": [2.4, 1.9, 1.7],
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.name == "train_loss"
    assert train.values == (2.3, 1.8, 1.4)
    assert validation.name == "validation_loss"
    assert validation.values == (2.4, 1.9, 1.7)


def test_extracts_pytorch_epoch_logs():
    series = extract_metric_series(
        {
            "epoch_logs": [
                {"epoch": 1, "train_loss": 2.0, "val_loss": 2.2},
                {"epoch": 2, "train_loss": 1.6, "val_loss": 1.8},
            ]
        }
    )

    train, validation = paired_metric(series, "loss")

    assert train is not None
    assert validation is not None
    assert train.values == (2.0, 1.6)
    assert train.steps == (1, 2)
    assert validation.values == (2.2, 1.8)
    assert validation.steps == (1, 2)


def test_normalizes_music_generation_metric_aliases():
    series = extract_metric_series(
        {
            "history": {
                "fad": [4.2, 3.8],
                "eval_clap_score": [0.24, 0.31],
                "eval_stft_loss": [0.72, 0.61],
            }
        }
    )

    assert series["frechet_audio_distance"].last == 3.8
    assert series["validation_clap_score"].split == "validation"
    assert series["validation_stft_loss"].last == 0.61
