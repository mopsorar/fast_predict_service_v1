from locust import HttpUser, constant_pacing, task


class FastPredictUser(HttpUser):
    wait_time = constant_pacing(0.1)

    @task
    def predict(self) -> None:
        with self.client.post(
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
        ) as response:
            if response.error is not None:
                response.failure(response.error)
                return
            if response.status_code != 200:
                response.failure(f"Unexpected HTTP status: {response.status_code}")
                return

            try:
                body = response.json()
            except ValueError:
                response.failure("Response is not valid JSON")
                return

            if not isinstance(body, dict) or body.get("prediction") != "setosa":
                response.failure("Unexpected prediction")
                return
            confidence = body.get("confidence")
            if type(confidence) not in (int, float) or not 0.0 <= confidence <= 1.0:
                response.failure("Invalid confidence")
