import importlib
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

import train_model as training
from app.model_loader import load_model, predict, predict_raw


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Pipeline, Path]:
    model_path = tmp_path_factory.mktemp("artifacts") / "model.pkl"
    return training.train_model(model_path), model_path


def test_model_save_and_load(trained_model: tuple[Pipeline, Path]) -> None:
    original, model_path = trained_model
    assert model_path.is_file()

    loaded = load_model(model_path)
    assert isinstance(loaded, Pipeline)
    features = [[5.1, 3.5, 1.4, 0.2], [6.5, 3.0, 5.8, 2.2]]
    assert loaded.predict(features).tolist() == original.predict(features).tolist()
    for actual, expected in zip(
        loaded.predict_proba(features), original.predict_proba(features), strict=True
    ):
        assert actual.tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize(
    ("features", "expected_class"),
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),
        ([6.0, 2.9, 4.5, 1.5], "versicolor"),
        ([6.5, 3.0, 5.8, 2.2], "virginica"),
    ],
)
def test_prediction_from_loaded_model(
    trained_model: tuple[Pipeline, Path], features: list[float], expected_class: str
) -> None:
    _, model_path = trained_model
    model = load_model(model_path)
    result = predict(model, features)

    assert set(result) == {"prediction", "confidence"}
    assert result["prediction"] == expected_class
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    class_index = list(model.classes_).index(expected_class)
    assert result["confidence"] == pytest.approx(
        model.predict_proba([features])[0, class_index]
    )


def test_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pkl")


def test_predict_raw(
    trained_model: tuple[Pipeline, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    model, model_path = trained_model
    features = [5.1, 3.5, 1.4, 0.2]
    result = predict_raw(features, model_path)

    assert result == predict(model, features)
    output = capsys.readouterr().out
    for label in ("Model load time:", "Inference time:"):
        line = next(line for line in output.splitlines() if line.startswith(label))
        assert float(line.removeprefix(label).removesuffix(" s")) >= 0.0


@pytest.mark.parametrize("features", [[], [1.0, 2.0, 3.0], [1.0] * 5])
def test_wrong_feature_count_raises(
    trained_model: tuple[Pipeline, Path], features: list[float]
) -> None:
    model, _ = trained_model
    with pytest.raises(ValueError, match="exactly four"):
        predict(model, features)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "invalid"])
def test_invalid_feature_raises(
    trained_model: tuple[Pipeline, Path], value: float | str
) -> None:
    model, _ = trained_model
    with pytest.raises(ValueError):
        predict(model, [value, 3.5, 1.4, 0.2])


def test_import_does_not_train(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_fit(*args: object, **kwargs: object) -> None:
        pytest.fail("Importing train_model must not train the model.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    importlib.reload(training)
