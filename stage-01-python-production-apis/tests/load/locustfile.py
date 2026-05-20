"""
Locust load test for Stage 01 API.

Run against a live server:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Or headless:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 10 --run-time 60s

Production targets from spec:
  - p99 framework overhead < 50ms
  - Container start < 2s
  - Image size < 250MB

Watch for:
  - p99 latency on /v1/items/ POST (most expensive path)
  - Error rate under sustained load
  - Memory growth over time (connection leak detection)
"""

import uuid

from locust import HttpUser, between, task


class ItemsAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        """Seed one item so GET requests have something to fetch."""
        r = self.client.post("/v1/items/", json={"name": "SeedItem", "price": 1.0})
        if r.status_code == 201:
            self.item_id = r.json()["id"]
        else:
            self.item_id = 1

    @task(3)
    def list_items(self):
        self.client.get("/v1/items/?limit=20", name="/v1/items/ [list]")

    @task(2)
    def get_item(self):
        self.client.get(f"/v1/items/{self.item_id}", name="/v1/items/{id} [get]")

    @task(1)
    def create_item(self):
        # Unique idempotency key per logical operation
        self.client.post(
            "/v1/items/",
            json={"name": f"LoadItem-{uuid.uuid4().hex[:8]}", "price": 9.99},
            headers={"Idempotency-Key": str(uuid.uuid4())},
            name="/v1/items/ [create]",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
