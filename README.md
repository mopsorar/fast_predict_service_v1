# FastPredict

Небольшой сервис классификации Iris на FastAPI. Модель обучается отдельно, сохраняется в `model.pkl` и загружается при старте приложения. Запросы к `/predict` используют уже загруженную модель.

В проекте сравниваются два способа выполнения inference — прямой вызов и thread pool, а также запуск с одним и четырьмя Uvicorn workers. Результаты нагрузочных тестов приведены ниже.

## Стек

- Python 3.11, FastAPI 0.115.12, Uvicorn 0.34.2, Pydantic 2.x.
- Scikit-Learn 1.6.1, joblib 1.4.2 — обучение и сохранение модели.
- uv — зависимости и окружение; версии зафиксированы в `uv.lock`.
- pytest, HTTPX, Ruff — тесты и проверки кода.
- Locust 2.47.0 — нагрузочное тестирование.
- Docker и GitHub Actions — сборка и проверка приложения.

Runtime-зависимости находятся в `dependencies` файла `pyproject.toml`, инструменты разработки — в группе `dev`. `requirements.txt` — экспорт окружения из `uv`, включая `dev`. Сам FastPredict как Python-пакет не устанавливается (`tool.uv.package = false`).

## Структура проекта

```text
app/
    main.py              # FastAPI, lifespan, middleware, маршруты
    model_loader.py      # Загрузка модели и предсказание
    schemas.py           # Валидация входа и выхода
    metrics.py           # Счётчик запросов и latency
train_model.py           # Обучение модели Iris
locustfile.py            # Сценарий нагрузочного тестирования
tests/                   # Тесты модели, API, метрик и Locust
artifacts/model.pkl      # Создаётся скриптом обучения
load_test/results/       # Сохранённая статистика Locust
Dockerfile
Makefile
pyproject.toml
uv.lock
requirements.txt
.github/workflows/ci.yml
```

## Запуск на Windows

Нужны Python 3.11 и [uv](https://docs.astral.sh/uv/). Команды выполняются из корня проекта в PowerShell.

```powershell
uv sync --locked
uv run --locked python train_model.py
$env:FASTPREDICT_EXECUTION_MODE = "threadpool"
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

После запуска:

- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`

Сервер останавливается через `Ctrl+C`. Активировать `.venv` вручную не обязательно: `uv run` использует окружение проекта.

### Обучение модели

`train_model.py` использует встроенный датасет Iris. Данные делятся на train/test (80/20) со стратификацией по классам и `random_state=42`. Модель — `Pipeline` из `StandardScaler` и `LogisticRegression(max_iter=1000)`.

Scaler обучается только на тренировочной выборке. После обучения скрипт выводит accuracy на тестовой выборке и сохраняет весь pipeline через `joblib.dump()` в `artifacts/model.pkl`.

При запуске API модель не переобучается. В `lifespan` она один раз загружается через `joblib.load()` и сохраняется в `app.state.model`. Каждый Uvicorn worker загружает собственную копию. Если файла нет, startup завершится ошибкой.

В `app/model_loader.py` есть отдельная функция `predict_raw()`, которая загружает модель для каждого вызова и печатает время загрузки и inference. Она нужна для демонстрации накладных расходов, а HTTP API её не использует:

```powershell
uv run --locked python -c "from app.model_loader import predict_raw; print(predict_raw([5.1, 3.5, 1.4, 0.2]))"
```

Файлы `joblib` нужно загружать только из доверенных источников.

## API

### `POST /predict`

Принимает четыре признака цветка в сантиметрах:

| Поле | Описание | Диапазон, см |
| --- | --- | --- |
| `sepal_length` | Длина чашелистика | `(0, 10]` |
| `sepal_width` | Ширина чашелистика | `(0, 5]` |
| `petal_length` | Длина лепестка | `(0, 10]` |
| `petal_width` | Ширина лепестка | `(0, 5]` |

Все поля обязательны. Строки вместо чисел, отрицательные и нулевые значения, `null`, `NaN`, бесконечности и выход за диапазон отклоняются с HTTP 422. Целые числа допустимы.

**Пример запроса:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Ответ HTTP 200:**

```json
{
  "prediction": "setosa",
  "confidence": 0.9808127381969128
}
```

`prediction` — один из классов `setosa`, `versicolor`, `virginica`; `confidence` — вероятность выбранного класса из `predict_proba()`.

Пример вызова из PowerShell:

```powershell
$body = @{
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

### `GET /health`

Проверка доступности приложения и наличия загруженной модели:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /metrics`

Статистика обработки HTTP-запросов в текущем worker:

```json
{
  "total_requests": 0,
  "average_latency_ms": 0.0
}
```

Здесь показано начальное состояние. Middleware считает все запросы, кроме самого `/metrics`, включая ответы 404/422 и необработанные ошибки. Время измеряется через `perf_counter()` вокруг `call_next()`, счётчик и накопленное время защищены `threading.Lock`.

Метрики не сохраняются между перезапусками и не объединяются между процессами. Это серверное время обработки до возврата ответа из `call_next()`, а не полное время HTTP-запроса со стороны клиента.

## Blocking, threadpool и workers

Режим выбирается через переменную окружения `FASTPREDICT_EXECUTION_MODE` при запуске приложения:

| Режим | Как выполняется `predict()` |
| --- | --- |
| `blocking` | Синхронно в потоке event loop |
| `threadpool` | Через `await run_in_threadpool(...)` в рабочем потоке |

По умолчанию используется `threadpool`. Для переключения нужно перезапустить Uvicorn:

```powershell
$env:FASTPREDICT_EXECUTION_MODE = "blocking"
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Thread pool позволяет не занимать поток event loop синхронным вычислением, но не гарантирует ускорение самого inference. На результат влияют накладные расходы потоков, GIL и работа нативных библиотек Scikit-Learn/NumPy.

Для запуска четырёх процессов вместо одного:

```powershell
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
```

У каждого worker свой event loop, модель и счётчики. Поэтому для суммарных показателей нескольких процессов используется статистика Locust, а не `/metrics`.

## Нагрузочное тестирование

Сценарий `locustfile.py` отправляет `POST /predict` с валидным примером Iris и проверяет HTTP-статус, JSON, ожидаемый класс и confidence. Для запросов установлен таймаут 10 секунд.

Условия четырёх экспериментов:

- 50 виртуальных пользователей, запуск по 50 пользователей/с;
- `constant_pacing(0.1)` — цель 10 запросов/с на пользователя, суммарно до 500 RPS;
- длительность — 60 секунд;
- Locust и Uvicorn запущены на одном Windows-компьютере;
- по одному прогону для каждой конфигурации.

Сервер запускается отдельно для нужного режима и количества workers. Затем в другом окне PowerShell:

```powershell
$case = "blocking_1"
$prefix = Join-Path "load_test/results" ($case + "_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force "load_test/results" | Out-Null

uv run --locked locust -f locustfile.py --headless --host http://127.0.0.1:8000 --users 50 --spawn-rate 50 --run-time 60s --csv $prefix --csv-full-history
```

Для остальных запусков изменяются режим Uvicorn, количество workers и имя `$case` (`threadpool_1`, `blocking_4`, `threadpool_4`). Если запросы не успевают завершаться за 100 мс, фактический RPS будет ниже целевого.

### Результаты

Итоговая статистика, выведенная Locust в консоль после завершения каждого запуска:

| Метрика | Blocking ×1 | Threadpool ×1 | Blocking ×4 | Threadpool ×4 |
| --- | ---: | ---: | ---: | ---: |
| RPS | 387.94 | 337.02 | 493.12 | 495.38 |
| Median, мс | 120 | 140 | 17 | 17 |
| P95, мс | 180 | 200 | 66 | 66 |
| Среднее, мс | 127 | 147 | 22 | 22 |
| Запросов | 23 184 | 20 147 | 29 462 | 29 592 |
| Ошибок | 0 | 0 | 0 | 0 |

На одном worker режим `blocking` показал более высокий RPS. С четырьмя workers оба режима приблизились к заданным 500 RPS, поэтому по этим данным нельзя сравнить их максимальную пропускную способность.

Это результаты единичных прогонов, а не универсальный benchmark. Измерения зависят от компьютера, фоновой нагрузки и конкуренции сервера с Locust за ресурсы. Клиентская latency Locust включает сетевые затраты и не равна времени одного inference.

В репозитории сохранены CSV с агрегированной статистикой. Locust пишет CSV периодически, поэтому счётчики и RPS в них местами отличаются от последнего вывода в консоли. Точную причину расхождений в этих запусках подтвердить не удалось; Median, P95 и число ошибок совпадают.

## Docker и Makefile

Dockerfile использует `python:3.11-slim` и uv 0.11.8. Runtime-зависимости устанавливаются командой `uv sync --locked --no-dev`, затем внутри сборки обучается модель. Приложение запускается от непривилегированного пользователя, по умолчанию с одним Uvicorn worker.

```powershell
docker build -t fastpredict:local .
docker run --rm -p 8000:8000 -e FASTPREDICT_EXECUTION_MODE=threadpool fastpredict:local
```

Количество workers можно изменить через `WEB_CONCURRENCY`:

```powershell
docker run --rm -p 8000:8000 -e FASTPREDICT_EXECUTION_MODE=blocking -e WEB_CONCURRENCY=4 fastpredict:local
```

Локально Docker не установлен. Сборка образа и smoke-тесты контейнера для обоих режимов **успешно прошли в GitHub Actions**: [запуск CI №40093613813](https://github.com/mopsorar/fast_predict_service_v1/actions/runs/40093613813).

В Makefile есть три цели:

| Команда | Назначение |
| --- | --- |
| `make train` | Обучение модели |
| `make run` | Локальный запуск API |
| `make test-load` | Locust: 50 пользователей, 60 секунд |

Например, для тестирования другого адреса:

```powershell
$csvPrefix = "load_test/results/manual_" + [guid]::NewGuid().ToString("N")
make test-load LOAD_HOST=http://127.0.0.1:8000 "CSV_PREFIX=$csvPrefix"
```

На Windows нужен отдельно установленный GNU Make. Без него все основные команды можно выполнить напрямую через `uv run`.

## Тесты и CI

Локальные проверки:

```powershell
uv run --locked python -m pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv pip check
```

В проекте **208 тестов**: загрузка и предсказания модели, Pydantic-валидация, lifespan, оба режима выполнения, метрики, обработка ошибок и сценарий Locust.

При открытии PR в `main` запускается [GitHub Actions workflow](.github/workflows/ci.yml):

- `checks` — тесты и Ruff;
- `docker-smoke` — сборка образа, `/health`, `/predict` и проверка HTTP 422 в режимах `blocking` и `threadpool`.

Оба задания прошли успешно для коммита `ac60a2f`. Нагрузочные тесты в CI не запускаются.

## Ограничения

- Сервис рассчитан на учебную модель Iris; результаты нагрузки не переносятся автоматически на другие модели и машины.
- Нет авторизации, rate limiting и общего хранилища метрик между workers.
- Горячей замены модели нет: обновление `model.pkl` требует перезапуска приложения.
- В используемой версии Starlette есть опубликованная проблема с обработкой заголовка `Host`. Обход метрик через этот заголовок исправлен на уровне приложения; обновление зависимости остаётся отдельной задачей.
