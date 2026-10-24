from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class InputData(BaseModel):
    model_config = ConfigDict(strict=True, allow_inf_nan=False)

    sepal_length: PositiveFloat = Field(
        le=10.0, description="Sepal length in centimeters."
    )
    sepal_width: PositiveFloat = Field(
        le=5.0, description="Sepal width in centimeters."
    )
    petal_length: PositiveFloat = Field(
        le=10.0, description="Petal length in centimeters."
    )
    petal_width: PositiveFloat = Field(
        le=5.0, description="Petal width in centimeters."
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
