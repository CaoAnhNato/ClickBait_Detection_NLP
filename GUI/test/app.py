from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from .model_registry import MODEL_REGISTRY
except ImportError:
    # Support running as a script: `python app.py` from GUI directory.
    from model_registry import MODEL_REGISTRY


def create_app(test_config: dict | None = None, model_service: Any | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config.update(TESTING=False)
    if test_config:
        app.config.update(test_config)

    if model_service is None:
        try:
            from .model_service import ClickbaitModelService
        except ImportError:
            # Support running as a script: `python app.py` from GUI directory.
            from model_service import ClickbaitModelService

        # Lazy loading keeps startup memory low; models load on first use.
        app.config["MODEL_SERVICE"] = ClickbaitModelService(preload_local_models=False, preload_orcd_model=True)
    else:
        app.config["MODEL_SERVICE"] = model_service

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model_options=[{"key": k, "label": v["label"]} for k, v in MODEL_REGISTRY.items()],
            selected_model="bart-mnli",
            input_text="",
            api_key_input="",
            custom_api_model_input="",
            custom_api_provider_input="",
            custom_api_base_input="",
            result=None,
            error=None,
        )

    @app.post("/")
    def detect_from_form():
        model_key = request.form.get("model", "")
        text = request.form.get("text", "")
        api_key = request.form.get("api_key", "")
        custom_api_model = request.form.get("custom_api_model", "")
        custom_api_provider = request.form.get("custom_api_provider", "")
        custom_api_base = request.form.get("custom_api_base", "")

        result = None
        error = None

        profile = MODEL_REGISTRY.get(model_key)
        if profile is None:
            error = "Invalid model"
        elif profile.get("backend") in {"gpt", "orcd"} and not str(api_key).strip():
            error = "API key is required for API-based methods (GPT/Gemini/Qwen/Llama/ORCD)"

        try:
            if error is None:
                result = app.config["MODEL_SERVICE"].predict(
                    text=text,
                    model_key=model_key,
                    api_key=api_key,
                    custom_api_model=custom_api_model,
                    custom_api_provider=custom_api_provider,
                    custom_api_base=custom_api_base,
                )
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

        return render_template(
            "index.html",
            model_options=[{"key": k, "label": v["label"]} for k, v in MODEL_REGISTRY.items()],
            selected_model=model_key or "bart-mnli",
            input_text=text,
            api_key_input=api_key,
            custom_api_model_input=custom_api_model,
            custom_api_provider_input=custom_api_provider,
            custom_api_base_input=custom_api_base,
            result=result,
            error=error,
        )

    @app.post("/api/detect")
    def detect_api():
        payload = request.get_json(silent=True) or {}
        model_key = (payload.get("model") or "").strip()
        text = payload.get("text") or ""
        api_key = payload.get("api_key") or ""
        custom_api_model = payload.get("api_model") or payload.get("custom_api_model") or ""
        custom_api_provider = payload.get("api_provider") or payload.get("custom_api_provider") or ""
        custom_api_base = payload.get("api_base") or payload.get("custom_api_base") or ""

        if model_key not in MODEL_REGISTRY:
            return jsonify({"error": "Invalid model"}), 400
        if not str(text).strip():
            return jsonify({"error": "Text must not be empty"}), 400
        if MODEL_REGISTRY[model_key].get("backend") in {"gpt", "orcd"} and not str(api_key).strip():
            return jsonify({"error": "API key is required for API-based methods (GPT/Gemini/Qwen/Llama/ORCD)"}), 400

        try:
            result = app.config["MODEL_SERVICE"].predict(
                text=text,
                model_key=model_key,
                api_key=api_key,
                custom_api_model=custom_api_model,
                custom_api_provider=custom_api_provider,
                custom_api_base=custom_api_base,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc)}), 500

        return jsonify(result), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
