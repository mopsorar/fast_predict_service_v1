import json
import sys
from collections.abc import Iterator
from pathlib import Path
from threading import get_ident
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

from app import main
from app.model_loader import Prediction, load_model
from app.schemas import InputData, PredictionResponse
from train_model import train_model


@pytest.fixture(scope="module")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("api_model") / "model.pkl"
    train_model(path)
    return path


@pytest.fixture(params=["blocking", "threadpool"])
def execution_mode(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    mode = request.param
    monkeypatch.setenv("FASTPREDICT_EXECUTION_MODE", mode)
    return mode


@pytest.fixture
def client(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
) -> Iterator[TestClient]:
    monkeypatch.setattr(main, "load_model", lambda: load_model(model_path))
    with TestClient(main.app) as test_client:
        assert main.app.state.execution_mode == execution_mode
        yield test_client


@pytest.fixture
def valid_payload() -> dict[str, float]:
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


def test_model_loaded_once_per_lifespan(
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, float],
    execution_mode: str,
) -> None:
    loader = Mock(side_effect=lambda: load_model(model_path))
    monkeypatch.setattr(main, "load_model", loader)

    for lifecycle in range(2):
        with TestClient(main.app) as test_client:
            model = main.app.state.model
            assert model is not None
            assert main.app.state.execution_mode == execution_mode
            assert loader.call_count == lifecycle + 1
            for _ in range(2):
                assert (
                    test_client.post("/predict", json=valid_payload).status_code == 200
                )
                assert test_client.get("/health").status_code == 200
            assert loader.call_count == lifecycle + 1
            assert main.app.state.model is model
        assert main.app.state.model is None


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), ValueError("corrupt")])
def test_startup_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch, error: Exception, execution_mode: str
) -> None:
    loader = Mock(side_effect=error)
    monkeypatch.setattr(main, "load_model", loader)
    with pytest.raises(type(error), match=str(error)):
        with TestClient(main.app):
            pytest.fail("Application must not start without a model.")
    loader.assert_called_once_with()


def test_default_execution_mode(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FASTPREDICT_EXECUTION_MODE", raising=False)
    monkeypatch.setattr(main, "load_model", lambda: load_model(model_path))
    with TestClient(main.app):
        assert main.app.state.execution_mode == "threadpool"


@pytest.mark.parametrize("mode", ["invalid", "", "THREADPOOL", " blocking"])
def test_invalid_execution_mode_fails_startup(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("FASTPREDICT_EXECUTION_MODE", mode)
    loader = Mock()
    monkeypatch.setattr(main, "load_model", loader)
    with pytest.raises(
        ValueError, match="FASTPREDICT_EXECUTION_MODE.*blocking.*threadpool"
    ):
        with TestClient(main.app):
            pytest.fail("Application must not start with an invalid execution mode.")
    loader.assert_not_called()


def test_execution_mode_is_not_reread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, valid_payload: dict[str, float]
) -> None:
    selected_mode = main.app.state.execution_mode
    monkeypatch.setenv("FASTPREDICT_EXECUTION_MODE", "invalid")
    assert client.post("/predict", json=valid_payload).status_code == 200
    assert main.app.state.execution_mode == selected_mode


def test_predict_execution_thread(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, valid_payload: dict[str, float]
) -> None:
    original_predict = main.predict
    prediction_threads: list[int] = []

    def recording_predict(model: Pipeline, features: list[float]) -> Prediction:
        prediction_threads.append(get_ident())
        return original_predict(model, features)

    pool = AsyncMock(wraps=main.run_in_threadpool)
    monkeypatch.setattr(main, "predict", recording_predict)
    monkeypatch.setattr(main, "run_in_threadpool", pool)
    assert client.portal is not None
    event_loop_thread = client.portal.call(get_ident)

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert len(prediction_threads) == 1
    if main.app.state.execution_mode == "threadpool":
        assert prediction_threads[0] != event_loop_thread
        pool.assert_awaited_once_with(
            recording_predict, main.app.state.model, list(valid_payload.values())
        )
    else:
        assert prediction_threads[0] == event_loop_thread
        pool.assert_not_awaited()


def test_execution_modes_return_identical_predictions(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, valid_payload: dict[str, float]
) -> None:
    model = load_model(model_path)
    loader = Mock(return_value=model)
    monkeypatch.setattr(main, "load_model", loader)
    results: list[list[dict[str, str | float]]] = []
    for mode in ("blocking", "threadpool"):
        monkeypatch.setenv("FASTPREDICT_EXECUTION_MODE", mode)
        with TestClient(main.app) as test_client:
            assert main.app.state.model is model
            mode_results = []
            for features in (
                [5.1, 3.5, 1.4, 0.2],
                [6.0, 2.9, 4.5, 1.5],
                [6.5, 3.0, 5.8, 2.2],
            ):
                payload = dict(zip(valid_payload, features, strict=True))
                response = test_client.post("/predict", json=payload)
                assert response.status_code == 200
                mode_results.append(response.json())
            results.append(mode_results)
        assert main.app.state.model is None
    assert results[0] == results[1]
    assert loader.call_count == 2


@pytest.mark.parametrize(
    ("features", "expected_class"),
    [
        ([5.1, 3.5, 1.4, 0.2], "setosa"),
        ([6.0, 2.9, 4.5, 1.5], "versicolor"),
        ([6.5, 3.0, 5.8, 2.2], "virginica"),
    ],
)
def test_predict(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    features: list[float],
    expected_class: str,
    valid_payload: dict[str, float],
) -> None:
    predictor = Mock(wraps=main.predict)
    monkeypatch.setattr(main, "predict", predictor)
    payload = dict(zip(valid_payload, features, strict=True))
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "confidence"}
    assert body["prediction"] == expected_class
    assert isinstance(body["confidence"], float)
    assert 0.0 <= body["confidence"] <= 1.0
    PredictionResponse.model_validate(body)
    predictor.assert_called_once_with(main.app.state.model, features)


def test_integer_features_are_valid(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"sepal_length": 5, "sepal_width": 3, "petal_length": 1, "petal_width": 1},
    )
    assert response.status_code == 200


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metrics_initial_state(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json() == {"total_requests": 0, "average_latency_ms": 0.0}
    assert type(response.json()["total_requests"]) is int
    assert type(response.json()["average_latency_ms"]) is float


def test_metrics_count_successful_requests(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=valid_payload).status_code == 200
    snapshot = client.get("/metrics").json()
    assert snapshot["total_requests"] == 2
    assert snapshot["average_latency_ms"] >= 0.0


def test_metrics_use_scope_path_with_malformed_host(
    client: TestClient, valid_payload: dict[str, float]
) -> None:
    response = client.post(
        "/predict", json=valid_payload, headers={"Host": "example.com/metrics?x="}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "setosa"
    snapshot = client.get("/metrics").json()
    assert snapshot["total_requests"] == 1
    assert client.get("/metrics").json() == snapshot


def test_metrics_count_validation_errors(client: TestClient) -> None:
    assert client.post("/predict", json={}).status_code == 422
    assert (
        client.post(
            "/predict", content="{", headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )
    assert client.get("/metrics").json()["total_requests"] == 2


def test_metrics_count_not_found(client: TestClient) -> None:
    assert client.get("/missing").status_code == 404
    assert client.get("/metrics").json()["total_requests"] == 1


def test_metrics_average_and_no_self_counting(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, valid_payload: dict[str, float]
) -> None:
    clock = Mock(side_effect=[1.0, 1.125, 5.0, 5.375])
    monkeypatch.setattr(main, "perf_counter", clock)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json=valid_payload).status_code == 200

    expected = {"total_requests": 2, "average_latency_ms": 250.0}
    for path in ("/metrics", "/metrics", "/metrics?view=current"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == expected
    assert clock.call_count == 4


@pytest.mark.parametrize("raise_server_exceptions", [True, False])
@pytest.mark.parametrize(
    "error", [RuntimeError("Inference failed"), ValueError("Invalid model state")]
)
def test_metrics_count_unhandled_exception(
    model_path: Path,
    execution_mode: str,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, float],
    raise_server_exceptions: bool,
    error: Exception,
) -> None:
    monkeypatch.setattr(main, "load_model", lambda: load_model(model_path))
    monkeypatch.setattr(main, "predict", Mock(side_effect=error))
    with TestClient(
        main.app, raise_server_exceptions=raise_server_exceptions
    ) as test_client:
        if raise_server_exceptions:
            with pytest.raises(type(error), match=str(error)):
                test_client.post("/predict", json=valid_payload)
        else:
            response = test_client.post("/predict", json=valid_payload)
            assert response.status_code == 500
            assert response.text == "Internal Server Error"
        assert test_client.get("/metrics").json()["total_requests"] == 1
        assert test_client.get("/health").status_code == 200
        assert test_client.get("/metrics").json()["total_requests"] == 2


def test_metrics_reset_per_lifespan(
    model_path: Path, execution_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "load_model", lambda: load_model(model_path))
    previous_metrics = None
    for _ in range(2):
        with TestClient(main.app) as test_client:
            assert main.app.state.metrics is not previous_metrics
            assert test_client.get("/metrics").json() == {
                "total_requests": 0,
                "average_latency_ms": 0.0,
            }
            assert test_client.get("/health").status_code == 200
            assert test_client.get("/metrics").json()["total_requests"] == 1
            previous_metrics = main.app.state.metrics


@pytest.mark.parametrize(
    "field", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
)
@pytest.mark.parametrize("value", [0, -1, "5.1", "invalid", True, None, [], {}])
def test_invalid_feature_returns_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, float],
    field: str,
    value: object,
) -> None:
    predictor = Mock(wraps=main.predict)
    monkeypatch.setattr(main, "predict", predictor)
    response = client.post("/predict", json={**valid_payload, field: value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
    predictor.assert_not_called()


@pytest.mark.parametrize(
    "field", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
)
def test_missing_feature_returns_422(
    client: TestClient, valid_payload: dict[str, float], field: str
) -> None:
    del valid_payload[field]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-Infinity", "1e309", "[NaN]", '{"nested": Infinity}']
)
def test_nonfinite_number_returns_422(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, float],
    value: str,
) -> None:
    predictor = Mock(wraps=main.predict)
    monkeypatch.setattr(main, "predict", predictor)
    payload = json.dumps(valid_payload).replace(
        '"sepal_length": 5.1', f'"sepal_length": {value}'
    )
    response = client.post(
        "/predict", content=payload, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "sepal_length"]
    predictor.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sepal_length", 10.01),
        ("sepal_width", 5.01),
        ("petal_length", 10.01),
        ("petal_width", 5.01),
        ("sepal_length", sys.float_info.max),
        ("sepal_width", sys.float_info.max),
        ("petal_length", sys.float_info.max),
        ("petal_width", sys.float_info.max),
    ],
)
def test_out_of_range_feature_returns_422_without_inference(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    valid_payload: dict[str, float],
    field: str,
    value: float,
) -> None:
    predictor = Mock(wraps=main.predict)
    monkeypatch.setattr(main, "predict", predictor)
    response = client.post("/predict", json={**valid_payload, field: value})

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", field]
    assert error["type"] == "less_than_equal"
    predictor.assert_not_called()


@pytest.mark.parametrize(
    ("field", "upper_bound"),
    [
        ("sepal_length", 10.0),
        ("sepal_width", 5.0),
        ("petal_length", 10.0),
        ("petal_width", 5.0),
    ],
)
def test_upper_bound_is_accepted(
    client: TestClient, valid_payload: dict[str, float], field: str, upper_bound: float
) -> None:
    response = client.post("/predict", json={**valid_payload, field: upper_bound})
    assert response.status_code == 200
    PredictionResponse.model_validate(response.json())


def test_all_iris_samples_are_within_input_bounds(
    valid_payload: dict[str, float],
) -> None:
    for sample in load_iris().data:
        InputData.model_validate(dict(zip(valid_payload, sample.tolist(), strict=True)))


@pytest.mark.parametrize("body", [b"\xff", b"private binary payload"])
def test_binary_body_returns_422_without_disclosing_input(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    predictor = Mock(wraps=main.predict)
    monkeypatch.setattr(main, "predict", predictor)
    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/octet-stream"}
    )

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body"]
    assert error["type"] == "model_attributes_type"
    assert error["input"] is None
    assert "UnicodeDecodeError" not in response.text
    assert "private binary payload" not in response.text
    predictor.assert_not_called()


def test_malformed_json_returns_422(client: TestClient) -> None:
    response = client.post(
        "/predict",
        content='{"sepal_length":',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "json_invalid"


@pytest.mark.parametrize("payload", [None, [], "invalid", 1])
def test_invalid_body_returns_422(client: TestClient, payload: object) -> None:
    response = client.post(
        "/predict",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_openapi(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    document = response.json()
    operation = document["paths"]["/predict"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/InputData"
    }
    assert operation["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/PredictionResponse"
    }
    assert "422" in operation["responses"]
    schemas = document["components"]["schemas"]
    upper_bounds = {
        "sepal_length": 10.0,
        "sepal_width": 5.0,
        "petal_length": 10.0,
        "petal_width": 5.0,
    }
    assert set(schemas["InputData"]["required"]) == set(upper_bounds)
    for field, upper_bound in upper_bounds.items():
        assert schemas["InputData"]["properties"][field]["type"] == "number"
        assert schemas["InputData"]["properties"][field]["exclusiveMinimum"] == 0
        assert schemas["InputData"]["properties"][field]["maximum"] == upper_bound
        assert "centimeters" in schemas["InputData"]["properties"][field]["description"]
    assert set(schemas["PredictionResponse"]["required"]) == {
        "prediction",
        "confidence",
    }
    confidence = schemas["PredictionResponse"]["properties"]["confidence"]
    assert confidence["minimum"] == 0
    assert confidence["maximum"] == 1
    assert "get" in document["paths"]["/health"]
