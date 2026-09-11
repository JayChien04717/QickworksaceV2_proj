"""Canonical automation catalog shared by the worker and workspace CLI."""

from QickworkspaceV2.device.models import RunDefaults
from QickworkspaceV2.data.serialization import digest


def catalog_payload(registry):
    entries = registry.catalog()
    for entry in entries:
        properties = entry["parameters"].setdefault("properties", {})
        if {"run_options", "request_id"} & properties.keys():
            raise ValueError("run_options and request_id are reserved worker arguments")
        properties.update({
            "run_options": RunDefaults.model_json_schema(),
            "request_id": {
                "type": "string", "minLength": 1, "maxLength": 128,
                "description": "Reuse for retries of this logical measurement only.",
            },
        })
    return {"experiments": entries, "revision": digest(sorted(entries, key=lambda entry: entry["id"]))}
