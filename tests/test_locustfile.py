from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import pytest

if TYPE_CHECKING:
    from locustfile import FastPredictUser


@pytest.fixture
def user(monkeypatch: pytest.MonkeyPatch) -> "FastPredictUser":
    # Keep Locust's monkey patching out of the application test process.
    monkeypatch.setenv("LOCUST_SKIP_MONKEY_PATCH", "1")
    from locust.env import Environment

    from locustfile import FastPredictUser

    monkeypatch.setattr(FastPredictUser, "host", "http://127.0.0.1:8000")
    return FastPredictUser(Environment())


@pytest.mark.parametrize(
    ("status", "body", "error", "failure_message"),
    [
        (200, {"prediction": "setosa", "confidence": 0.98}, None, None),
        (200, {"prediction": "setosa", "confidence": 0.0}, None, None),
        (200, {"prediction": "setosa", "confidence": 1.0}, None, None),
        (0, None, TimeoutError("request timed out"), "request timed out"),
        (422, {"detail": []}, None, "Unexpected HTTP status: 422"),
        (500, None, None, "Unexpected HTTP status: 500"),
        (307, None, None, "Unexpected HTTP status: 307"),
        (200, [], None, "Unexpected prediction"),
        (200, {}, None, "Unexpected prediction"),
        (
            200,
            {"prediction": "virginica", "confidence": 0.98},
            None,
            "Unexpected prediction",
        ),
        (200, {"prediction": "setosa"}, None, "Invalid confidence"),
        (200, {"prediction": "setosa", "confidence": True}, None, "Invalid confidence"),
        (
            200,
            {"prediction": "setosa", "confidence": "0.98"},
            None,
            "Invalid confidence",
        ),
        (200, {"prediction": "setosa", "confidence": -0.1}, None, "Invalid confidence"),
        (200, {"prediction": "setosa", "confidence": 1.1}, None, "Invalid confidence"),
        (
            200,
            {"prediction": "setosa", "confidence": float("nan")},
            None,
            "Invalid confidence",
        ),
        (
            200,
            {"prediction": "setosa", "confidence": float("inf")},
            None,
            "Invalid confidence",
        ),
    ],
)
def test_predict_response_validation(
    user: "FastPredictUser",
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: object,
    error: Exception | None,
    failure_message: str | None,
) -> None:
    response = MagicMock(status_code=status, error=error)
    response.__enter__.return_value = response
    response.json.return_value = body
    post = Mock(return_value=response)
    monkeypatch.setattr(user.client, "post", post)

    user.predict()

    post.assert_called_once_with(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
        catch_response=True,
        allow_redirects=False,
        timeout=10.0,
    )
    if failure_message is None:
        response.failure.assert_not_called()
    else:
        response.failure.assert_called_once()
        assert str(response.failure.call_args.args[0]) == failure_message


def test_invalid_json_is_reported(
    user: "FastPredictUser", monkeypatch: pytest.MonkeyPatch
) -> None:
    response = MagicMock(status_code=200, error=None)
    response.__enter__.return_value = response
    response.json.side_effect = ValueError("invalid JSON")
    monkeypatch.setattr(user.client, "post", Mock(return_value=response))

    user.predict()

    response.failure.assert_called_once_with("Response is not valid JSON")
