# import pytest
# from api.app import app
# from fastapi.testclient import TestClient
# from api.config import settings
#
#
# @pytest.fixture
# def client():
#     api_key = settings.api_key
#
#     if not api_key:
#         raise RuntimeError("API_KEY is not configured in the test container")
#
#     with TestClient(
#         app,
#         headers={"x-api-key": api_key},
#     ) as test_client:
#         yield test_client


from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from langchain_community.vectorstores import FAISS

from api.app import app
from api.dependencies import get_vector_store


@pytest.fixture
def mock_vector_store() -> Generator[MagicMock, None, None]:
    mock_db = MagicMock(spec=FAISS)

    # Replace the real dependency
    app.dependency_overrides[get_vector_store] = lambda: mock_db

    yield mock_db

    # Restore original dependency
    app.dependency_overrides.clear()
