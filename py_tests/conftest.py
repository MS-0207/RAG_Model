import pytest
from api.app import app
from fastapi.testclient import TestClient
from api.config import settings


@pytest.fixture
def client():
    api_key = settings.api_key

    if not api_key:
        raise RuntimeError("API_KEY is not configured in the test container")

    with TestClient(
        app,
        headers={"x-api-key": api_key},
    ) as test_client:
        yield test_client
