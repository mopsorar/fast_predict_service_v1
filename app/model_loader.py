from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import joblib
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "model.pkl"


class Prediction(TypedDict):
    prediction: str
    confidence: float


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trusted, locally trained model artifact."""
    return joblib.load(model_path)


def predict(model: Pipeline, input_data: Sequence[float]) -> Prediction:
    """Predict from sepal length/width and petal length/width, in centimeters."""
    if len(input_data) != 4:
        raise ValueError("Expected exactly four Iris features.")

    probabilities = model.predict_proba([input_data])[0]
    class_index = int(probabilities.argmax())
    return {
        "prediction": str(model.classes_[class_index]),
        "confidence": float(probabilities[class_index]),
    }


def predict_raw(
    input_data: Sequence[float], model_path: str | Path = DEFAULT_MODEL_PATH
) -> Prediction:
    load_started = perf_counter()
    model = load_model(model_path)
    load_seconds = perf_counter() - load_started

    inference_started = perf_counter()
    result = predict(model, input_data)
    inference_seconds = perf_counter() - inference_started

    print(f"Model load time: {load_seconds:.6f} s")
    print(f"Inference time: {inference_seconds:.6f} s")
    return result
