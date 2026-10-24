import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from math import isfinite
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from app.metrics import HTTPMetrics
from app.model_loader import load_model, predict
from app.schemas import InputData, PredictionResponse


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    execution_mode = os.getenv("FASTPREDICT_EXECUTION_MODE", "threadpool")
    if execution_mode not in {"blocking", "threadpool"}:
        raise ValueError(
            "FASTPREDICT_EXECUTION_MODE must be 'blocking' or 'threadpool', "
            f"got {execution_mode!r}."
        )

    application.state.model = load_model()
    application.state.execution_mode = execution_mode
    application.state.metrics = HTTPMetrics()
    try:
        yield
    finally:
        application.state.model = None


app = FastAPI(title="FastPredict", lifespan=lifespan)


@app.middleware("http")
async def collect_http_metrics(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    if request.scope["path"] == "/metrics":
        return await call_next(request)

    metrics = request.app.state.metrics
    started = perf_counter()
    try:
        return await call_next(request)
    finally:
        metrics.record(perf_counter() - started)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Keep errors JSON-safe without decoding or exposing binary request bodies.
    errors = jsonable_encoder(
        exc.errors(),
        custom_encoder={
            float: lambda value: value if isfinite(value) else str(value),
            bytes: lambda _value: None,
        },
    )
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    input_data: InputData, request: Request
) -> PredictionResponse:
    features = [
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
    ]
    model = request.app.state.model
    if request.app.state.execution_mode == "threadpool":
        result = await run_in_threadpool(predict, model, features)
    else:
        result = predict(model, features)
    return PredictionResponse(**result)


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": request.app.state.model is not None}


@app.get("/metrics")
async def metrics(request: Request) -> dict[str, int | float]:
    return request.app.state.metrics.snapshot()
