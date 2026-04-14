from __future__ import annotations

import os
import re
import sys
import threading
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import AppSettings


class ORCDPredictService:
    def __init__(self, settings: AppSettings):
        self._settings = settings
        self._predict_lock = threading.RLock()
        self._service = self._build_reference_service()

    def _build_reference_service(self) -> Any:
        gui_test_dir = self._settings.gui_test_dir
        if not gui_test_dir.exists():
            raise FileNotFoundError(f"Missing reference directory: {gui_test_dir}")

        gui_test_dir_str = str(gui_test_dir)
        if gui_test_dir_str not in sys.path:
            sys.path.insert(0, gui_test_dir_str)

        module_path = Path(gui_test_dir) / "model_service.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Missing reference file: {module_path}")

        self._install_litellm_compat_if_needed()

        try:
            spec = importlib.util.spec_from_file_location("orcd_reference_model_service", str(module_path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot create import spec for {module_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ClickbaitModelService = getattr(module, "ClickbaitModelService")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to import ClickbaitModelService from {gui_test_dir}: {exc}") from exc

        local_weights_root = self._settings.workspace_root / "Bert_Fami" / "weights"
        service = ClickbaitModelService(
            weights_root=local_weights_root,
            preload_local_models=False,
            preload_orcd_model=False,
        )

        # Pin ORCD paths explicitly so backend always points to local checkpoint.
        service._orcd_base_dir = self._settings.orcd_base_dir
        service._orcd_weight_path = self._settings.orcd_weight_path

        return service

    @staticmethod
    def _install_litellm_compat_if_needed() -> None:
        if "litellm" in sys.modules:
            return

        try:
            import litellm  # noqa: F401
            return
        except Exception:
            # Some Python 3.9 environments fail importing modern litellm/aiohttp typing internals.
            sys.modules.pop("litellm", None)

        try:
            from . import litellm_compat
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load litellm compatibility shim: {exc}") from exc

        sys.modules["litellm"] = litellm_compat

    @staticmethod
    def _extract_api_key(raw_value: str) -> str:
        text = (raw_value or "").strip()
        if not text:
            return ""

        # Prefer OpenAI-compatible keys when mixed content is present.
        for pattern in (
            r"sk-[A-Za-z0-9_-]{10,}",
            r"xai-[A-Za-z0-9_-]{10,}",
            r"hf_[A-Za-z0-9]{10,}",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(0)

        for line in text.splitlines():
            candidate = line.strip()
            if not candidate or candidate.startswith("#"):
                continue

            if ":" in candidate:
                candidate = candidate.split(":", 1)[1].strip()
            elif "=" in candidate:
                candidate = candidate.split("=", 1)[1].strip()

            candidate = candidate.strip().strip("\"'")
            if candidate and all(ch not in candidate for ch in "\r\n"):
                return candidate.split()[0]

        return ""

    def _resolve_api_key(self) -> str:
        env_key = self._extract_api_key(os.getenv(self._settings.api_key_env_var, ""))
        if env_key:
            return env_key

        if self._settings.api_key_file.exists():
            file_key = self._extract_api_key(self._settings.api_key_file.read_text(encoding="utf-8"))
            if file_key:
                return file_key

        raise RuntimeError(
            "Missing ORCD API key. Set ORCD_API_KEY or provide a non-empty api_key.txt at workspace root."
        )

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", (title or "").strip())

    @staticmethod
    def _is_api_backend_model(model_key: str) -> bool:
        lowered = (model_key or "").strip().lower()
        return lowered.startswith("orcd") or lowered.startswith("gpt") or lowered.startswith("gemini") or lowered.startswith("qwen") or lowered.startswith("llama")

    def _invoke_predict(self, normalized_title: str, model_key: str) -> dict[str, Any]:
        needs_api_key = self._is_api_backend_model(model_key)
        api_key = self._resolve_api_key() if needs_api_key else ""

        with self._predict_lock:
            return self._service.predict(
                text=normalized_title,
                model_key=model_key,
                api_key=api_key,
                custom_api_model=(self._settings.api_model_override if needs_api_key else ""),
                custom_api_provider=(self._settings.api_provider_override if needs_api_key else ""),
                custom_api_base=(self._settings.api_base_override if needs_api_key else ""),
            )

    def _should_fallback_to_local(self, primary_model_key: str, fallback_model_key: str, exc: Exception) -> bool:
        if not self._settings.enable_api_fallback:
            return False

        if not primary_model_key or not fallback_model_key:
            return False

        if primary_model_key == fallback_model_key:
            return False

        if not self._is_api_backend_model(primary_model_key):
            return False

        if self._is_api_backend_model(fallback_model_key):
            return False

        message = str(exc).lower()
        error_markers = (
            "connection error",
            "failed to fetch",
            "timed out",
            "timeout",
            "connection refused",
            "connection reset",
            "network",
            "dns",
            "name resolution",
            "temporary failure",
            "service unavailable",
            "api key",
        )
        return any(marker in message for marker in error_markers)

    @lru_cache(maxsize=1024)
    def _predict_cached(self, normalized_title: str) -> dict[str, Any]:
        model_key = str(self._settings.model_key).strip()
        if not model_key:
            raise RuntimeError("ORCD_MODEL_KEY is empty. Configure a valid model key.")

        fallback_model_key = str(self._settings.fallback_local_model_key).strip()
        effective_model_key = model_key

        try:
            result = self._invoke_predict(normalized_title, model_key)
        except Exception as primary_exc:  # noqa: BLE001
            if not self._should_fallback_to_local(model_key, fallback_model_key, primary_exc):
                raise

            try:
                result = self._invoke_predict(normalized_title, fallback_model_key)
                effective_model_key = fallback_model_key
            except Exception as fallback_exc:  # noqa: BLE001
                raise RuntimeError(
                    "Primary prediction failed and local fallback also failed. "
                    f"primary={model_key}, fallback={fallback_model_key}, "
                    f"primary_error={primary_exc}, fallback_error={fallback_exc}"
                ) from fallback_exc

        label = int(result.get("label", 0))
        confidence = float(result.get("confidence", 0.0)) * 100.0

        return {
            "is_clickbait": label == 1,
            "confidence": round(max(0.0, min(100.0, confidence)), 2),
            "label": label,
            "model": str(result.get("model", effective_model_key)),
            "device": str(self._service.device),
        }

    def predict(self, title: str) -> dict[str, Any]:
        normalized_title = self._normalize_title(title)
        if not normalized_title:
            raise ValueError("title must not be empty")

        hits_before = self._predict_cached.cache_info().hits
        result = dict(self._predict_cached(normalized_title))
        hits_after = self._predict_cached.cache_info().hits
        result["cached"] = hits_after > hits_before
        return result

    def health_snapshot(self) -> dict[str, Any]:
        cache_info = self._predict_cached.cache_info()
        return {
            "status": "ok",
            "device": str(self._service.device),
            "model": self._settings.model_key,
            "cache": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "maxsize": cache_info.maxsize or 0,
                "currsize": cache_info.currsize,
            },
        }

    def clear_cache(self) -> None:
        self._predict_cached.cache_clear()
