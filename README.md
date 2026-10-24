# FastPredict

FastPredict — небольшой сервис классификации Iris на FastAPI. Модель обучается отдельно, сохраняется через joblib и загружается при запуске приложения. HTTP-запросы используют уже загруженный объект модели.

Проект демонстрирует работу с lifespan, валидацию входных данных, различия между прямым CPU-bound вызовом и выполнением через thread pool, влияние нескольких Uvicorn workers и измерение HTTP latency под нагрузкой.

## Технологический стек

Целевая версия Python — 3.11; локальные эксперименты выполнены на Python 3.11.15 под Windows. Для управления окружением используется uv; в Docker и CI зафиксирован uv 0.11.8.

| Компонент | Версия / назначение |
| --- | --- |
| FastAPI | 0.115.12 — HTTP API и lifespan |
| Pydantic | 2.x — входные данные и ответы; точная версия находится в `uv.lock` |
| Uvicorn | 0.34.2 — ASGI-сервер и worker-процессы |
| Scikit-Learn | 1.6.1 — Iris, preprocessing и LogisticRegression |
| joblib | 1.4.2 — сохранение и загрузка модели |
| Locust | 2.47.0 — HTTP-нагрузка и клиентская статистика |
| pytest / HTTPX | 9.1.1 / 0.28.1 — тесты и TestClient |
| Ruff | 0.11.8 — lint и форматирование |

Прямые зависимости описаны в `pyproject.toml`, полный граф с версиями и хешами — в `uv.lock`. Проект настроен с `tool.uv.package = false`: устанавливаются зависимости, без установки FastPredict как Python-пакета.

Основные `dependencies` содержат только FastAPI, Uvicorn, Scikit-Learn и joblib для API и обучения. pytest, HTTPX, Ruff и Locust находятся в dependency group `dev`, которая включается uv по умолчанию при локальной работе. Docker устанавливает только основные зависимости через `--no-dev`; CI явно включает `dev`.

`requirements.txt` — полный экспорт окружения из uv, включая группу `dev`, а не независимый источник версий. Повторный экспорт:

```powershell
uv export --locked --group dev --format requirements-txt --no-emit-project --output-file requirements.txt
```

## Структура проекта

```text
app/
    __init__.py
    main.py              # FastAPI, lifespan, execution mode и HTTP middleware
    model_loader.py      # Загрузка модели, predict и predict_raw
    schemas.py           # Pydantic-схемы
    metrics.py           # Статистика HTTP-запросов и Lock
artifacts/
    model.pkl            # Артефакт, создаваемый обучением
tests/
    test_model.py
    test_api.py
    test_metrics.py
    test_locustfile.py
load_test/
    results/             # CSV результатов Locust
.github/
    workflows/
        ci.yml
train_model.py
locustfile.py
pyproject.toml
uv.lock
requirements.txt
Dockerfile
.dockerignore
Makefile
README.md
```

## Локальный запуск: Windows PowerShell

Требуются Python 3.11 и uv, доступные в `PATH`. Все команды выполняются из корня проекта.

```powershell
uv sync --locked
uv run --locked python train_model.py
$env:FASTPREDICT_EXECUTION_MODE = "threadpool"
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

`uv sync --locked` синхронизирует проектное `.venv` с lock-файлом. Существующее окружение переиспользуется; вручную активировать его для `uv run` не требуется.

После startup API доступен на `http://127.0.0.1:8000`, Swagger UI — на `http://127.0.0.1:8000/docs`, OpenAPI — на `http://127.0.0.1:8000/openapi.json`. Для остановки сервера используется `Ctrl+C`.

Во время lifespan модель загружается один раз в `app.state.model`, конфигурация проверяется и создаётся статистика запросов. Ошибка загрузки модели прерывает startup. При shutdown ссылка на модель сбрасывается. Обучение при запуске API не выполняется: сначала нужен `artifacts/model.pkl`.

## Обучение и базовое предсказание

```powershell
uv run --locked python train_model.py
```

Скрипт использует встроенный Iris и Pipeline `StandardScaler → LogisticRegression`. Данные делятся на train/test в пропорции 80/20 со стратификацией по классам. Для разбиения и классификатора задан `random_state=42`; `max_iter=1000`.

Scaler обучается только на train-выборке и сохраняется вместе с классификатором. После обучения скрипт печатает версию Scikit-Learn, accuracy на test-выборке и путь сохранения. Вызов `joblib.dump()` создаёт или перезаписывает `artifacts/model.pkl`. Импорт `train_model.py` обучение не запускает.

Обучение и загрузка используют Scikit-Learn 1.6.1 из общего lock-файла. Артефакты joblib следует загружать только из доверенного источника.

В `app/model_loader.py` разделены три функции:

- `load_model()` загружает Pipeline через `joblib.load()`;
- `predict(model, input_data)` выполняет inference на переданном объекте;
- `predict_raw(input_data)` демонстрирует загрузку из файла и inference, отдельно печатая время обеих операций через `perf_counter()`.

```powershell
uv run --locked python -c "from app.model_loader import predict_raw; print(predict_raw([5.1, 3.5, 1.4, 0.2]))"
```

`predict_raw()` не используется в HTTP-обработчике, поскольку загружает файл при каждом вызове.

## HTTP API

### POST /predict

Четыре признака передаются именованными полями, значения измеряются в сантиметрах:

| Поле | Признак | Допустимый диапазон, см | Позиция при передаче модели |
| --- | --- | --- | --- |
| `sepal_length` | Длина чашелистика | `(0, 10]` | 1 |
| `sepal_width` | Ширина чашелистика | `(0, 5]` | 2 |
| `petal_length` | Длина лепестка | `(0, 10]` | 3 |
| `petal_width` | Ширина лепестка | `(0, 5]` | 4 |

Все поля обязательны. Разрешены конечные числовые значения в указанных диапазонах, включая целые числа. Это прикладные ограничения, покрывающие встроенный датасет Iris с запасом; границы также указаны в OpenAPI. Строки с числами, `bool`, `null`, `NaN` и бесконечности отклоняются. Значения вне диапазонов возвращают HTTP 422 до вызова inference.

Запрос:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Пример ответа HTTP 200 для сохранённой модели:

```json
{
  "prediction": "setosa",
  "confidence": 0.9808127381969128
}
```

`prediction` — название класса: `setosa`, `versicolor` или `virginica`. `confidence` — вероятность выбранного класса из `predict_proba()`, в диапазоне от 0 до 1.

В другом окне PowerShell:

```powershell
$body = @{
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body
```

Неполный или синтаксически неверный JSON, отрицательные и другие невалидные значения возвращают HTTP 422 со списком ошибок в `detail`. Неконечные числа в описании ошибки преобразуются в строки, чтобы сам ответ 422 оставался корректным JSON.

### GET /health

При работающем приложении с загруженной моделью возвращается HTTP 200:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

### GET /metrics

Возвращает количество обработанных запросов и среднее серверное время обработки в миллисекундах. До первого учитываемого запроса ответ имеет вид:

```json
{
  "total_requests": 0,
  "average_latency_ms": 0.0
}
```

Это начальное состояние, а не результат измерения нагрузки.

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/metrics"
```

Middleware учитывает успешные запросы, `/health`, ошибки 422/404 и запросы с необработанными исключениями. Обновление выполняется в `finally`, без перехвата исходной ошибки. Запросы к `/metrics` не меняют статистику.

Внутри `HTTPMetrics` хранятся счётчик и сумма длительностей в секундах. Среднее вычисляется как `total_seconds × 1000 / total_requests`; при нулевом счётчике возвращается `0.0`. `threading.Lock` защищает короткие операции обновления и получения согласованного снимка. Во время `await` и ML-предсказания Lock не удерживается. Новый lifespan создаёт новую статистику.

Важно различать границы измерений:

| Измерение | Что включает |
| --- | --- |
| Время загрузки в `predict_raw()` | Загрузку и десериализацию файла модели |
| Время inference в `predict_raw()` | Вызов `predict()` на загруженной модели |
| `average_latency_ms` | Время вокруг `call_next()` в middleware, включая валидацию и ожидание thread pool, до возврата объекта ответа |
| Locust response time | Клиентское время HTTP-запроса с сетью и получением ответа |

Серверная метрика не включает полную передачу тела ответа клиенту и не равна клиентской HTTP latency.

## Режимы выполнения inference

Режим выбирается через `FASTPREDICT_EXECUTION_MODE` при startup и сохраняется в `app.state.execution_mode`. На каждом запросе переменная окружения не перечитывается. Значения чувствительны к регистру; неизвестное или пустое значение вызывает ошибку запуска.

| Режим | Выполнение `predict()` |
| --- | --- |
| `blocking` | Прямой синхронный вызов внутри async endpoint, в потоке event loop |
| `threadpool` | `await starlette.concurrency.run_in_threadpool(...)`, в рабочем потоке |

По умолчанию используется `threadpool`. В обоих режимах используются одна функция предсказания и загруженный при startup объект модели; JSON-контракт одинаков.

Перед запуском сервера выберите режим:

```powershell
$env:FASTPREDICT_EXECUTION_MODE = "blocking"
```

или:

```powershell
$env:FASTPREDICT_EXECUTION_MODE = "threadpool"
```

Для изменения режима работающий сервер нужно остановить и запустить заново.

В режиме `blocking` вычисление занимает поток event loop. Thread pool выносит вызов из этого потока, но добавляет накладные расходы и не гарантирует увеличение throughput. Для CPU-bound Python-кода важен GIL; поведение нативных операций Scikit-Learn/NumPy также влияет на результат.

## Uvicorn workers

Worker — отдельный процесс приложения со своим event loop, экземпляром модели и HTTPMetrics. Несколько workers позволяют распределять запросы между процессами, но увеличивают потребление памяти. Это другой механизм конкурентности, чем thread pool внутри одного процесса.

Например, запуск с четырьмя workers:

```powershell
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
```

Каждый worker самостоятельно выполняет lifespan и загружает модель один раз. Счётчики `/metrics` не агрегируются: запрос к этому endpoint показывает состояние только того процесса, который его обработал. Для сравнения конфигураций с несколькими workers используется агрегированная статистика Locust.

## Нагрузочное тестирование Locust

`locustfile.py` содержит один класс `FastPredictUser` и одну задачу: POST `/predict` с признаками из примера API. `/health` и `/metrics` в сценарии не вызываются. Редиректы отключены, в HTTP-запрос передаётся `timeout=10.0`.

Через `catch_response=True` проверяются HTTP 200, корректность JSON, ожидаемый класс `setosa` и числовой confidence от 0 до 1. Неожиданные статусы, неверные ответы и ошибки соединения учитываются как failures.

Параметры сравнения: 50 пользователей, spawn rate 50 пользователей/с, `constant_pacing(0.1)`, длительность 60 секунд. Один пользователь старается выполнять одну задачу каждые 100 мс; при одном запросе в задаче цель составляет 10 RPS на пользователя и 500 RPS суммарно. Если задача выполняется дольше 100 мс, пользователь ждёт её окончания и фактический RPS снижается.

Проверка загрузки сценария без генерации нагрузки:

```powershell
uv run --locked locust -f locustfile.py --list
```

Для каждого эксперимента используйте два окна PowerShell. В первом задайте режим и количество workers по таблице, затем запустите сервер:

| Эксперимент | `FASTPREDICT_EXECUTION_MODE` | `$workers` | `$case` для CSV |
| --- | --- | --- | --- |
| Blocking ×1 | `blocking` | 1 | `blocking_1` |
| Threadpool ×1 | `threadpool` | 1 | `threadpool_1` |
| Blocking ×4 | `blocking` | 4 | `blocking_4` |
| Threadpool ×4 | `threadpool` | 4 | `threadpool_4` |

```powershell
$env:FASTPREDICT_EXECUTION_MODE = "blocking"
$workers = 1
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers $workers
```

Дождитесь startup всех workers. Во втором окне выберите соответствующий `$case` и выполните Locust:

```powershell
$case = "blocking_1"
$prefix = Join-Path "load_test/results" ($case + "_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force "load_test/results" | Out-Null

uv run --locked locust -f locustfile.py --headless --host http://127.0.0.1:8000 --users 50 --spawn-rate 50 --run-time 60s --csv $prefix --csv-full-history
```

Между экспериментами остановите предыдущий Uvicorn через `Ctrl+C`. Меняются только execution mode, количество workers и имя CSV-префикса; данные, темп и продолжительность остаются одинаковыми. UUID в префиксе позволяет повторять запуски без перезаписи предыдущих файлов.

Locust сохраняет `<prefix>_stats.csv`, `_stats_history.csv`, `_failures.csv` и `_exceptions.csv`. В итоговом `_stats.csv` используйте строку `Aggregated`:

| Показатель | Поле CSV |
| --- | --- |
| Фактический RPS | `Requests/s` |
| Median, мс | `Median Response Time` |
| P95, мс | `95%` |
| Average, мс | `Average Response Time` |
| Количество запросов | `Request Count` |
| Количество ошибок | `Failure Count` |
| Процент ошибок | `100 × Failure Count / Request Count`; при нуле запросов — `0.0` |

## Результаты benchmarks

Условия ручных экспериментов: 50 виртуальных пользователей, `constant_pacing(0.1)`, цель 500 RPS, длительность каждого запуска 60 секунд. Locust и сервер работали на одном Windows-компьютере. Для сравнения использован один прогон на конфигурацию; конфигурация оборудования не зафиксирована.

Источник таблицы — финальный консольный вывод Locust, предоставленный пользователем.

| Метрика | Blocking ×1 | Threadpool ×1 | Blocking ×4 | Threadpool ×4 |
| --- | --- | --- | --- | --- |
| RPS | 387.94 | 337.02 | 493.12 | 495.38 |
| Median, ms | 120 | 140 | 17 | 17 |
| P95, ms | 180 | 200 | 66 | 66 |
| Average, ms | 127 | 147 | 22 | 22 |
| Requests | 23184 | 20147 | 29462 | 29592 |
| Failures | 0 | 0 | 0 | 0 |

CSV сохраняются периодически и могут содержать более ранний снимок статистики, чем финальный консольный вывод. Это возможное объяснение расхождений, но причина конкретных различий между доступными CSV и консольными результатами не доказана. Для основной benchmark-таблицы используется указанная выше финальная консольная статистика.

### Интерпретация

- В этом одиночном эксперименте threadpool с одним worker показал меньший RPS, чем blocking. Из этого не следует универсальное преимущество прямого вызова.
- Четыре Uvicorn workers существенно улучшили показатели обоих режимов относительно одного worker.
- Оба режима с четырьмя workers практически достигли заданного потолка нагрузки 500 RPS. Это не доказывает равенство их максимальной пропускной способности: нагрузка выше этого уровня не исследовалась.
- Один прогон на конфигурацию не позволяет оценить разброс результатов или статистическую значимость различий.
- Общий Windows-компьютер для клиента и сервера означает конкуренцию за ресурсы. Результаты не переносятся автоматически на другую машину или Docker.
- Нулевое количество ошибок относится только к этим прогонам и не доказывает отсутствие ошибок при иной нагрузке.

60 секунд включают разгон пользователей. Запросы, не завершённые к остановке Locust, могут не попасть в итоговую статистику. Значения response time относятся к клиентской HTTP latency, а не к чистому времени ML-inference или среднему из `/metrics`.

## Docker

Dockerfile использует `python:3.11-slim` и uv 0.11.8. Сначала копируются `pyproject.toml` и `uv.lock` и выполняется `uv sync --locked --no-dev`; этот слой кешируется независимо от изменений исходников. Инструменты из группы `dev` в образ не устанавливаются. Затем копируется код и запускается `python train_model.py`.

Таким образом, `artifacts/model.pkl` формируется внутри сборки, а обучение и загрузка используют одно окружение со Scikit-Learn 1.6.1. `.dockerignore` допускает только необходимые исходники и файлы установки: локальные артефакты, `.venv`, Git, кеши, файлы окружения и результаты Locust в контекст не попадают.

Приложение запускается от пользователя `app` с UID 10001, слушает `0.0.0.0:8000`. По умолчанию `WEB_CONCURRENCY=1`, режим inference по умолчанию — `threadpool`.

Команды для среды с доступным Docker:

```powershell
docker build -t fastpredict:local .
docker run --rm -p 8000:8000 -e FASTPREDICT_EXECUTION_MODE=threadpool fastpredict:local
```

После остановки предыдущего контейнера можно запустить blocking с четырьмя workers:

```powershell
docker run --rm -p 8000:8000 -e FASTPREDICT_EXECUTION_MODE=blocking -e WEB_CONCURRENCY=4 fastpredict:local
```

Docker локально недоступен: успешная сборка и работа контейнера пока не подтверждены. Docker smoke tests настроены в GitHub Actions; их успешное выполнение также пока не подтверждено.

## Makefile

Makefile требует GNU Make и uv в `PATH`. В Windows GNU Make может отсутствовать; все основные операции доступны напрямую через приведённые выше команды PowerShell с `uv run`.

| Команда | Действие |
| --- | --- |
| `make train` | Обучение и сохранение модели |
| `make run` | Запуск Uvicorn на `127.0.0.1:8000` |
| `make test-load` | Headless Locust: 50 пользователей, spawn rate 50, 60 секунд |

Для нагрузки можно изменить адрес сервера и CSV-префикс:

```powershell
$csvPrefix = "load_test/results/manual_" + [guid]::NewGuid().ToString("N")
make test-load LOAD_HOST=http://127.0.0.1:8000 "CSV_PREFIX=$csvPrefix"
```

По умолчанию Makefile формирует `load_test/results/locust_<UUID>`. Рецепты не используют Unix-утилиты. Локальное выполнение целей через GNU Make не подтверждено: GNU Make в текущей среде недоступен.

## Тесты и проверки качества

```powershell
uv run --locked python -m pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv pip check
```

Тесты покрывают сохранение и загрузку модели, предсказания трёх классов, confidence, ошибки входных данных и отсутствие обучения при импорте. API проверяется через TestClient с lifespan в обоих execution mode, включая реальный поток выполнения, однократную загрузку модели и ошибки startup.

Отдельно проверяются начальное состояние метрик, средняя latency, конкурентные обновления, исключение `/metrics` из подсчёта, сброс статистики и учёт необработанных исключений. Тесты Locust проверяют обработку ответов и ошибок без запуска полноценной нагрузки.

Последний подтверждённый локальный запуск: 208 тестов прошли, Ruff lint и format check прошли, конфликтов зависимостей не обнаружено. В зависимостях остаются `DeprecationWarning` Starlette/AnyIO и Scikit-Learn/SciPy. Локальные результаты не означают успешного выполнения CI.

## GitHub Actions

Workflow [.github/workflows/ci.yml](.github/workflows/ci.yml) настроен для pull request в `main` и содержит два job на `ubuntu-latest`:

- `checks`: Python 3.11, uv 0.11.8, `uv sync --locked --group dev`, все pytest-тесты, Ruff lint и format check;
- `docker-smoke`: сборка образа, запуск с одним worker последовательно в threadpool и blocking, до 30 проверок готовности `/health`, проверка HTTP 200 и JSON для `/health` и `/predict`, проверка HTTP 422 для отрицательного признака.

После успешной проверки режима контейнер останавливается и удаляется. При ошибке выводятся его логи; отдельный шаг с `if: always()` очищает оставшийся контейнер. Провал проверки должен завершать job с ошибкой. Полноценные нагрузочные тесты, деплой и публикация образов в CI не настроены.

Workflow проверен статически, включая синтаксис shell-скриптов и встроенного Python. Успешный фактический запуск GitHub Actions и Docker smoke tests пока не подтверждён.

## Ограничения реализации

- Это небольшой учебный inference-сервис с одной моделью Iris; результаты не характеризуют произвольные ML-нагрузки.
- Blocking-режим намеренно выполняет синхронное вычисление в event loop. Thread pool не гарантирует ускорения CPU-bound кода.
- Модель и метрики загружаются и хранятся отдельно в каждом worker; общего межпроцессного хранилища статистики нет.
- Метрики сбрасываются при новом lifespan и не сохраняются между перезапусками.
- Модель загружается при startup; автоматического обучения или горячей замены модели в API нет.
- Аутентификация, ограничение частоты запросов и дополнительные производственные механизмы эксплуатации не реализованы.
- Benchmark ограничен одним прогоном на конфигурацию, 50 пользователями, длительностью 60 секунд, общим компьютером клиента и сервера и целевой нагрузкой 500 RPS.
- Работа Docker и CI требует подтверждения реальным запуском на GitHub Actions runner.
