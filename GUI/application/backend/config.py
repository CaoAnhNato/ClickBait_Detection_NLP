from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    workspace_root: Path
    gui_test_dir: Path
    orcd_base_dir: Path
    orcd_weight_path: Path
    placeholder_model_dir: Path
    api_key_env_var: str
    api_key_file: Path
    model_key: str
    fallback_local_model_key: str
    enable_api_fallback: bool
    api_model_override: str
    api_provider_override: str
    api_base_override: str
    host: str
    port: int
    cors_allow_origins: tuple[str, ...]


def _parse_bool_env(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_path(raw_path: str, workspace_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (workspace_root / candidate).resolve()


def load_settings() -> AppSettings:
    workspace_root = Path(__file__).resolve().parents[3]

    gui_test_dir = _resolve_path("GUI/test", workspace_root)
    orcd_base_dir = _resolve_path("ORCD/GPT_3.5", workspace_root)
    orcd_weight_path = _resolve_path(
        os.getenv("ORCD_WEIGHT_PATH", "ORCD/GPT_3.5/weight/best_teachermodel.pth"),
        workspace_root,
    )

    placeholder_model_dir = _resolve_path(
        os.getenv("BERT_MODEL_PATH", "./my_bert_model"),
        workspace_root,
    )

    api_key_file = _resolve_path(
        os.getenv("ORCD_API_KEY_FILE", "api_key.txt"),
        workspace_root,
    )

    origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = tuple(part.strip() for part in origins_raw.split(",") if part.strip())
    if not origins:
        origins = ("*",)

    return AppSettings(
        workspace_root=workspace_root,
        gui_test_dir=gui_test_dir,
        orcd_base_dir=orcd_base_dir,
        orcd_weight_path=orcd_weight_path,
        placeholder_model_dir=placeholder_model_dir,
        api_key_env_var=os.getenv("ORCD_API_KEY_ENV", "ORCD_API_KEY"),
        api_key_file=api_key_file,
        model_key=os.getenv("ORCD_MODEL_KEY", "gemini-zero"),
        fallback_local_model_key=os.getenv("ORCD_FALLBACK_LOCAL_MODEL_KEY", "bart-mnli"),
        enable_api_fallback=_parse_bool_env(os.getenv("ORCD_ENABLE_API_FALLBACK"), True),
        api_model_override=os.getenv("ORCD_API_MODEL_OVERRIDE", "gemini-2.5-flash").strip(),
        api_provider_override=os.getenv("ORCD_API_PROVIDER_OVERRIDE", "gemini").strip(),
        api_base_override=os.getenv("ORCD_API_BASE_OVERRIDE", "").strip(),
        host=os.getenv("CLICKBAIT_BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("CLICKBAIT_BACKEND_PORT", "8000")),
        cors_allow_origins=origins,
    )
