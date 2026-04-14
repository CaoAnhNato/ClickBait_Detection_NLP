with open('test_model_service.py', 'r') as f:
    lines = f.readlines()

new_lines = [
"""from pathlib import Path
from types import SimpleNamespace

import pytest

from GUI.test.model_registry import MODEL_REGISTRY
from GUI.test.model_service import ClickbaitModelService


def test_available_models_has_exact_three_keys():
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    assert set(service.available_models().keys()) == {
        "bart-mnli",
        "roberta",
        "ggbert",
        "sheepdog",
        "gpt4o-zero",
        "gpt4o-few",
        "orcd-gpt35",
        "orcd-gpt4o",
        "gemini-zero",
        "qwen-zero",
        "llama-zero",
    }


@pytest.mark.parametrize("model_key", ["bart-mnli", "roberta", "ggbert"])
def test_predict_label_is_binary(monkeypatch, model_key):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    monkeypatch.setattr(service, "load_model", lambda _: None)
    monkeypatch.setattr(
        service,
        "_predict_with_loaded_model",
        lambda _1, _2, _3: {"label": 1, "confidence": 0.8},
    )

    result = service.predict("headline", model_key)
    assert result["label"] in (0, 1)
    assert result["model"] == model_key
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_empty_text_raises():
"""
]
for i, line in enumerate(lines):
    if "def test_predict_empty_text_raises():" in line:
        new_lines.append("".join(lines[i+1:]))
        break

with open('test_model_service.py', 'w') as f:
    f.write(new_lines[0] + new_lines[1])
