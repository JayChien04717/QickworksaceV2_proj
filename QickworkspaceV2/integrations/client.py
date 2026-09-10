"""Thin worker client with explicit async job handles and bounded waits."""

from __future__ import annotations
from uuid import uuid4
import time
import httpx


class WorkerClient:
    def __init__(self, url="http://127.0.0.1:8000", *, client=None):
        self.url = url.rstrip("/")
        self.client = client or httpx.Client(timeout=30)

    def _request(self, method, path, **kwargs):
        response = self.client.request(method, self.url + path, **kwargs)
        response.raise_for_status()
        return response.json()

    def submit(self, experiment, *, parameters=None, run_options=None, request_id=None):
        return self._request(
            "POST",
            "/experiments/run",
            json={
                "experiment": experiment,
                "parameters": parameters or {},
                "run_options": run_options or {},
                "request_id": request_id or uuid4().hex,
            },
        )

    def status(self, job_id):
        return self._request("GET", f"/experiments/{job_id}")

    def cancel(self, job_id):
        return self._request("POST", f"/experiments/{job_id}/cancel")

    def wait(self, job_id, *, timeout=240, poll_interval=0.25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.status(job_id)
            if job["status"] == "completed":
                return self._request("GET", f"/experiments/{job_id}/result")
            if job["status"] in ("failed", "cancelled", "interrupted"):
                return {
                    "status": "failed",
                    "error": job["error"] or job["status"],
                    "metadata": {"job_id": job_id, "run_id": job["run_id"]},
                }
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Job {job_id} is still owned by the worker; inspect status or request cooperative cancellation"
        )

    def run(self, experiment, *, parameters=None, run_options=None, request_id=None, timeout=240):
        job = self.submit(experiment, parameters=parameters, run_options=run_options, request_id=request_id)
        return self.wait(job["id"], timeout=timeout)

    def close(self):
        self.client.close()
