.PHONY: train run test-load

UV ?= uv
LOAD_HOST ?= http://127.0.0.1:8000
CSV_PREFIX ?= load_test/results/locust_$(shell $(UV) run --locked python -c "from uuid import uuid4; print(uuid4().hex)")

train:
	$(UV) run --locked python train_model.py

run:
	$(UV) run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000

test-load:
	$(UV) run --locked locust -f locustfile.py --headless --host "$(LOAD_HOST)" --users 50 --spawn-rate 50 --run-time 60s --csv "$(CSV_PREFIX)" --csv-full-history
