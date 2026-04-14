from pathlib import Path
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
        lambda _1, _2: {"label": 1, "confidence": 0.8},
    )

    result = service.predict("headline", model_key)
    assert result["label"] in (0, 1)
    assert result["model"] == model_key
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_empty_text_raises():
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    with pytest.raises(ValueError, match="Text must not be empty"):
        service.predict("   ", "bart-mnli")


def test_predict_invalid_model_raises():
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    with pytest.raises(ValueError, match="Unsupported model"):
        service.predict("abc", "invalid")


def test_switching_model_reloads(monkeypatch):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    load_calls = []

    def fake_load(model_key):
        load_calls.append(model_key)

    monkeypatch.setattr(service, "load_model", fake_load)
    monkeypatch.setattr(
        service,
        "_predict_with_loaded_model",
        lambda *_args, **_kwargs: {"label": 0, "confidence": 0.88},
    )

    service.predict("a", "bart-mnli")
    service.predict("b", "roberta")

    assert load_calls == ["bart-mnli", "roberta"]


def test_local_model_switch_clears_previous_cache(monkeypatch):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    service._local_models = {"bart-mnli": {"model": object(), "tokenizer": object()}}

    clear_calls = {"n": 0}
    load_calls = []

    def fake_clear():
        clear_calls["n"] += 1
        service._local_models.clear()

    def fake_load(model_key):
        load_calls.append(model_key)
        service._local_models[model_key] = {"model": object(), "tokenizer": object()}

    monkeypatch.setattr(service, "_clear_local_cache", fake_clear)
    monkeypatch.setattr(service, "_load_local_model", fake_load)

    service.load_model("roberta")
    assert clear_calls["n"] == 1
    assert load_calls == ["roberta"]


def test_default_constructor_is_lazy_load():
    service = ClickbaitModelService(weights_root=Path("/tmp"))
    assert service._local_models == {}
    assert service._orcd_model_bundle is None


def test_oom_falls_back_to_cpu(monkeypatch):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)

    class FakeDevice:
        type = "cuda"

    service._device = FakeDevice()
    call_count = {"n": 0}

    def fake_load(_):
        return None

    def fake_predict(_text, _model_key):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("CUDA out of memory")
        return {"label": 1, "confidence": 0.91}

    monkeypatch.setattr(service, "load_model", fake_load)
    monkeypatch.setattr(service, "_predict_with_loaded_model", fake_predict)
    monkeypatch.setattr(service, "_clear_local_cache", lambda: None)
    monkeypatch.setattr(service, "preload_all_local_models", lambda: None)

    from GUI.test import model_service as ms

    monkeypatch.setattr(ms, "torch", type("TorchStub", (), {"device": staticmethod(lambda *_: "cpu")}))

    result = service.predict("oom test", "ggbert")
    assert result["label"] == 1
    assert call_count["n"] == 2


def test_gpt_requires_api_key():
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    with pytest.raises(ValueError, match="API key is required for API-based models"):
        service.predict("headline", "gpt4o-zero")


def test_orcd_requires_api_key():
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    with pytest.raises(ValueError, match="API key is required for ORCD methods"):
        service.predict("headline", "orcd-gpt35")


def test_missing_dependencies_raise(monkeypatch):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    from GUI.test import model_service as ms

    monkeypatch.setattr(ms, "torch", None)
    with pytest.raises(RuntimeError, match="Missing dependencies"):
        service.load_model("bart-mnli")


def test_registry_points_to_expected_weight_dirs():
    assert MODEL_REGISTRY["bart-mnli"]["weights_dir"] == "facebook_bart-large-mnli"
    assert MODEL_REGISTRY["roberta"]["weights_dir"] == "FacebookAI_roberta-large"
    assert MODEL_REGISTRY["ggbert"]["weights_dir"] == "google-bert_bert-large-cased-whole-word-masking"
    assert MODEL_REGISTRY["gpt4o-zero"]["backend"] == "gpt"
    assert MODEL_REGISTRY["gpt4o-few"]["backend"] == "gpt"
    assert MODEL_REGISTRY["orcd-gpt35"]["backend"] == "orcd"
    assert MODEL_REGISTRY["orcd-gpt4o"]["backend"] == "orcd"


def test_orcd_weight_path_resolved_from_workspace_root():
    fake_weights = Path("/tmp/project/Bert_Fami/weights")
    service = ClickbaitModelService(weights_root=fake_weights, preload_local_models=False)

    assert service._workspace_root == Path("/tmp/project")
    assert service._orcd_base_dir == Path("/tmp/project/ORCD/GPT_3.5")
    assert service._orcd_weight_path == Path("/tmp/project/ORCD/GPT_3.5/weight/best_teachermodel.pth")


def test_sheepdog_path_resolved_from_workspace_root():
    fake_weights = Path("/tmp/project/Bert_Fami/weights")
    service = ClickbaitModelService(weights_root=fake_weights, preload_local_models=False)

    assert service._workspace_root == Path("/tmp/project")
    assert service._sheepdog_base_dir == Path("/tmp/project/SheepDog")


def test_predict_sheepdog_uses_backend(monkeypatch):
    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    load_calls = []

    def fake_load(model_key):
        load_calls.append(model_key)

    monkeypatch.setattr(service, "load_model", fake_load)
    monkeypatch.setattr(
        service,
        "_predict_with_sheepdog",
        lambda *_args, **_kwargs: {"label": 1, "confidence": 0.9, "model": "sheepdog"},
    )

    result = service.predict("headline", "sheepdog")
    assert result["label"] == 1
    assert result["model"] == "sheepdog"
    assert load_calls == ["sheepdog"]


def test_orcd_disagree_regenerate_rescore_uses_score_prompt(monkeypatch):
    from GUI.test import model_service as ms

    captured_prompts = []
    disagree_rescore_calls = {"n": 0}

    def fake_completion(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        captured_prompts.append(prompt)

        if "score the title's content" in prompt:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="50"))])
        if "make people believe the title" in prompt:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[agree reason]"))])
        if "make people disbelieve the title" in prompt:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[disagree reason]"))])

        if "Re-score based on the title content" in prompt and "The agree reasoning content is" in prompt:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="65"))])

        if "Re-score based on the title content" in prompt and "The disagree reasoning content is" in prompt:
            disagree_rescore_calls["n"] += 1
            score = "48" if disagree_rescore_calls["n"] == 1 else "35"
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=score))])

        if "Analyze the disagree reasoning" in prompt:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[disagree analysis]"))])
        if "Regenerate disagree reasoning" in prompt:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="[new disagree reason] (improved logic)"))]
            )

        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="50"))])

    monkeypatch.setattr(ms, "completion", fake_completion)

    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    result = service._run_orcd_reasoning_generation("A sample title", "orcd-gpt4o",
        "gemini-zero",
        "qwen-zero",
        "llama-zero", "fake-key")

    disagree_gate = result["flow_trace"]["disagree_gate"]
    assert disagree_gate["passed"] is True
    assert len(disagree_gate["iterations"]) == 2

    regenerate_disagree_prompt_count = sum("Regenerate disagree reasoning" in p for p in captured_prompts)
    assert regenerate_disagree_prompt_count == 1
    assert disagree_rescore_calls["n"] == 2


def test_predict_gpt_uses_custom_api_model(monkeypatch):
    from GUI.test import model_service as ms

    captured_models = []

    def fake_completion(**kwargs):
        captured_models.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Yes"))])

    monkeypatch.setattr(ms, "completion", fake_completion)

    service = ClickbaitModelService(weights_root=Path("/tmp"), preload_local_models=False)
    result = service.predict(
        "A sample title",
        "gpt4o-zero",
        api_key="sk-test",
        custom_api_model="gpt-4o-mini",
    )

    assert captured_models[-1] == "gpt-4o-mini"
    assert result["api_model_used"] == "gpt-4o-mini"
