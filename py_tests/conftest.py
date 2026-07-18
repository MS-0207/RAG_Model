import os

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture
def client():
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError("API_KEY is not configured in the test container")

    with TestClient(
        app,
        headers={"x-api-key": api_key},
    ) as test_client:
        yield test_client