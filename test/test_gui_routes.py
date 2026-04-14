import pytest


@pytest.mark.parametrize("model_key", ["bart-mnli", "roberta", "ggbert", "sheepdog"])
def test_index_post_detect_success(client, fake_service, model_key):
    response = client.post(
        "/",
        data={"model": model_key, "text": "This is a sample headline"},
    )

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Predicted Label" in page
    assert model_key in page
    assert fake_service.calls[-1]["model_key"] == model_key


@pytest.mark.parametrize("model_key", ["gpt4o-zero", "gpt4o-few", "orcd-gpt35", "orcd-gpt4o"])
def test_index_post_detect_success_api_models(client, fake_service, model_key):
    response = client.post(
        "/",
        data={
            "model": model_key,
            "text": "This is a sample headline",
            "api_key": "sk-test",
            "custom_api_model": "gpt-4o-mini",
        },
    )

    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Predicted Label" in page
    assert model_key in page
    assert fake_service.calls[-1]["model_key"] == model_key
    assert fake_service.calls[-1]["api_key"] == "sk-test"
    assert fake_service.calls[-1]["custom_api_model"] == "gpt-4o-mini"


def test_index_get_renders_page(client):
    response = client.get("/")
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Clickbait Detection" in page
    assert "Detect" in page


def test_api_detect_success(client):
    response = client.post(
        "/api/detect",
        json={"model": "bart-mnli", "text": "Breaking: something happened"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label"] in (0, 1)
    assert payload["model"] == "bart-mnli"


def test_api_detect_success_sheepdog(client):
    response = client.post(
        "/api/detect",
        json={"model": "sheepdog", "text": "Breaking: something happened"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label"] in (0, 1)
    assert payload["model"] == "sheepdog"


@pytest.mark.parametrize("model_key", ["gpt4o-zero", "gpt4o-few", "orcd-gpt35", "orcd-gpt4o"])
def test_api_detect_success_api_models(client, model_key):
    response = client.post(
        "/api/detect",
        json={
            "model": model_key,
            "text": "Breaking: something happened",
            "api_key": "sk-test",
            "api_model": "gpt-4o-mini",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["label"] in (0, 1)
    assert payload["model"] == model_key


def test_api_detect_invalid_model(client):
    response = client.post(
        "/api/detect",
        json={"model": "unknown", "text": "hello"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Invalid model"


def test_api_detect_empty_text(client):
    response = client.post(
        "/api/detect",
        json={"model": "bart-mnli", "text": "   "},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "Text must not be empty"


def test_api_detect_service_exception(client, app):
    class BoomService:
        def predict(
            self,
            text,
            model_key,
            api_key=None,
            custom_api_model=None,
            custom_api_provider=None,
            custom_api_base=None,
        ):
            raise RuntimeError("inference failed")

    app.config["MODEL_SERVICE"] = BoomService()
    response = client.post(
        "/api/detect",
        json={"model": "bart-mnli", "text": "abc"},
    )
    assert response.status_code == 500
    assert "inference failed" in response.get_json()["error"]


def test_api_detect_gpt_requires_api_key(client):
    response = client.post(
        "/api/detect",
        json={"model": "gpt4o-zero", "text": "Breaking title"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "API key is required for API-based methods (GPT/Gemini/Qwen/Llama/ORCD)"


def test_api_detect_orcd_requires_api_key(client):
    response = client.post(
        "/api/detect",
        json={"model": "orcd-gpt35", "text": "Breaking title"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "API key is required for API-based methods (GPT/Gemini/Qwen/Llama/ORCD)"
