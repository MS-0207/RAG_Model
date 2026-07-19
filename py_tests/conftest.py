from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_community.vectorstores import FAISS

from api.app import app
from api.dependencies import get_vector_store


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def mock_vector_store() -> Generator[MagicMock, None, None]:
    mock_db = MagicMock(spec=FAISS)

    app.dependency_overrides[get_vector_store] = lambda: mock_db

    yield mock_db

    app.dependency_overrides.pop(get_vector_store, None)