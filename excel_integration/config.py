from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationSettings:
    service_url: str


def load_settings() -> IntegrationSettings:
    """Where the add-in finds `DV_Service`. The queue settings that used to live here went with
    `worker.py` and `queue_clients.py`: the service is the worker and the queue now."""
    return IntegrationSettings(
        service_url=os.getenv("RF_SERVICE_URL", "http://127.0.0.1:8000"),
    )
