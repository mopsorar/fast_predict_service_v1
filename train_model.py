from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model_loader import DEFAULT_MODEL_PATH


def train_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    iris = load_iris()
    labels = iris.target_names[iris.target]
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Scikit-Learn version: {sklearn.__version__}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    return model


if __name__ == "__main__":
    train_model()
