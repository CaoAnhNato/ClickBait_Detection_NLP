import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GUI.test.app import create_app  # noqa: E402


class FakeService:
    def __init__(self):
        self.calls = []

    def predict(
        self,
        text,
        model_key,
        api_key=None,
        custom_api_model=None,
        custom_api_provider=None,
        custom_api_base=None,
    ):
        self.calls.append(
            {
                "text": text,
                "model_key": model_key,
                "api_key": api_key,
                "custom_api_model": custom_api_model,
                "custom_api_provider": custom_api_provider,
                "custom_api_base": custom_api_base,
            }
        )
        return {
            "label": 1,
            "confidence": 0.99,
            "model": model_key,
            "api_model_used": custom_api_model or "",
        }


@pytest.fixture
def fake_service():
    return FakeService()


@pytest.fixture
def app(fake_service):
    return create_app(test_config={"TESTING": True}, model_service=fake_service)


@pytest.fixture
def client(app):
    return app.test_client()
