from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import DataLoader as _TorchDataLoader
except ImportError:
    torch = None
    _TorchDataset = None
    _TorchDataLoader = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, RobertaModel, RobertaTokenizer
    from transformers.utils import logging
    logging.set_verbosity_error()
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    BertTokenizer = None
    RobertaModel = None
    RobertaTokenizer = None

try:
    from litellm import completion
    import litellm
    litellm.suppress_debug_info = True
except ImportError:
    completion = None

try:
    from .model_registry import MODEL_REGISTRY, get_weights_root
except ImportError:
    from model_registry import MODEL_REGISTRY, get_weights_root


class ClickbaitModelService:
    def __init__(
        self,
        weights_root: Optional[Path] = None,
        preload_local_models: bool = False,
        preload_orcd_model: bool = False,
    ):
        self.weights_root = weights_root or get_weights_root()
        self._workspace_root = self.weights_root.parents[1] if len(self.weights_root.parents) > 1 else self.weights_root.parent
        if torch is None:
            self._device = "cpu"
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._local_models: Dict[str, Dict[str, object]] = {}
        self._few_shot_cache: Dict[int, List[Dict[str, str]]] = {}
        self._orcd_base_dir = self._workspace_root / "ORCD" / "GPT_3.5"
        self._orcd_weight_path = self._orcd_base_dir / "weight" / "best_teachermodel.pth"
        self._orcd_model_bundle: Optional[Dict[str, Any]] = None
        self._orcd_tokenizer: Optional[Any] = None
        self._orcd_max_len = 100
        self._sheepdog_base_dir = self._workspace_root / "SheepDog"
        self._sheepdog_model_bundle: Optional[Dict[str, Any]] = None
        self._sheepdog_max_len = 512

        if preload_local_models:
            self.preload_all_local_models()
        if preload_orcd_model:
            self.preload_orcd_model()

    @property
    def device(self) -> str:
        return str(self._device)

    def available_models(self) -> Dict[str, str]:
        return {k: v["label"] for k, v in MODEL_REGISTRY.items()}

    def _local_model_keys(self) -> List[str]:
        return [key for key, meta in MODEL_REGISTRY.items() if meta.get("backend") == "local"]

    def _gpt_model_keys(self) -> List[str]:
        return [key for key, meta in MODEL_REGISTRY.items() if meta.get("backend") == "gpt"]

    def _orcd_model_keys(self) -> List[str]:
        return [key for key, meta in MODEL_REGISTRY.items() if meta.get("backend") == "orcd"]

    def _sheepdog_model_keys(self) -> List[str]:
        return [key for key, meta in MODEL_REGISTRY.items() if meta.get("backend") == "sheepdog"]

    def _get_model_dir(self, model_key: str) -> Path:
        profile = MODEL_REGISTRY.get(model_key)
        if profile is None:
            raise ValueError(f"Unsupported model: {model_key}")

        model_dir = self.weights_root / profile["weights_dir"]
        if not (model_dir / "config.json").exists() or not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Missing model files in {model_dir}")
        return model_dir

    def _clear_local_cache(self) -> None:
        self._local_models.clear()
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clear_sheepdog_cache(self) -> None:
        self._sheepdog_model_bundle = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_local_model(self, model_key: str) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("Missing dependencies: install torch and transformers to run inference")

        if model_key in self._local_models:
            return

        model_dir = self._get_model_dir(model_key)

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model.to(self._device)
        model.eval()

        self._local_models[model_key] = {
            "tokenizer": tokenizer,
            "model": model,
        }

    def preload_all_local_models(self) -> None:
        local_keys = self._local_model_keys()
        if not local_keys:
            return

        try:
            for key in local_keys:
                self._load_local_model(key)
        except RuntimeError as exc:
            if torch is not None and getattr(self._device, "type", "cpu") == "cuda" and "out of memory" in str(exc).lower():
                self._device = torch.device("cpu")
                self._clear_local_cache()
                for key in local_keys:
                    self._load_local_model(key)
            else:
                raise

    def preload_orcd_model(self) -> None:
        if not self._orcd_model_keys():
            return
        try:
            self._ensure_orcd_model_loaded()
        except Exception:
            # Keep app startup resilient; ORCD errors surface on demand at predict-time.
            pass

    def load_model(self, model_key: str) -> None:
        profile = MODEL_REGISTRY.get(model_key)
        if profile is None:
            raise ValueError(f"Unsupported model: {model_key}")

        if profile.get("backend") == "local":
            # Keep memory footprint predictable by holding one local transformer at a time.
            if model_key not in self._local_models and self._local_models:
                self._clear_local_cache()
            self._load_local_model(model_key)
            return

        if profile.get("backend") == "gpt":
            return

        if profile.get("backend") == "orcd":
            self._ensure_orcd_model_loaded()
            return

        if profile.get("backend") == "sheepdog":
            self._ensure_sheepdog_model_loaded(model_key)
            return

        if profile.get("backend") == "generate_and_predict":
            self._ensure_orcd_model_loaded()
            return

        raise ValueError(f"Unknown backend for model: {model_key}")

    def _find_latest_sheepdog_checkpoint(self, model_key: str) -> Path:
        profile = MODEL_REGISTRY[model_key]
        checkpoint_pattern = profile.get("checkpoint_pattern", "clickbait_iter*.m")
        checkpoint_dir = self._sheepdog_base_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Missing SheepDog checkpoint directory: {checkpoint_dir}")

        candidates = list(checkpoint_dir.glob(str(checkpoint_pattern)))
        if not candidates:
            raise FileNotFoundError(
                f"No SheepDog checkpoint matches pattern '{checkpoint_pattern}' in {checkpoint_dir}"
            )

        def extract_iter(path: Path) -> int:
            match = re.search(r"iter(\d+)", path.name)
            if match:
                return int(match.group(1))
            return -1

        return sorted(candidates, key=extract_iter)[-1]

    def _build_sheepdog_model(self):
        if torch is None or RobertaModel is None:
            raise RuntimeError("Missing dependencies: install torch and transformers to run SheepDog inference")

        import torch.nn as nn

        class SheepDogRobertaClassifier(nn.Module):
            def __init__(self, n_classes: int):
                super().__init__()
                self.roberta = RobertaModel.from_pretrained("roberta-base")
                self.dropout = nn.Dropout(p=0.5)
                self.fc_out = nn.Linear(self.roberta.config.hidden_size, n_classes)
                self.binary_transform = nn.Linear(self.roberta.config.hidden_size, 2)

            def forward(self, input_ids, attention_mask):
                outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                pooled_outputs = outputs[1]
                pooled_outputs = self.dropout(pooled_outputs)
                output = self.fc_out(pooled_outputs)
                binary_output = self.binary_transform(pooled_outputs)
                return output, binary_output

        return SheepDogRobertaClassifier(n_classes=4)

    def _ensure_sheepdog_model_loaded(self, model_key: str) -> None:
        if self._sheepdog_model_bundle is not None:
            return
        if torch is None or RobertaTokenizer is None:
            raise RuntimeError("Missing dependencies: install torch and transformers to run SheepDog inference")

        checkpoint_path = self._find_latest_sheepdog_checkpoint(model_key)
        model = self._build_sheepdog_model().to(self._device)

        checkpoint = torch.load(str(checkpoint_path), map_location=self._device)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Invalid SheepDog checkpoint format: expected state_dict dict at {checkpoint_path}")

        if any(str(k).startswith("module.") for k in checkpoint.keys()):
            checkpoint = {str(k).replace("module.", "", 1): v for k, v in checkpoint.items()}

        try:
            model.load_state_dict(checkpoint, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to load SheepDog checkpoint {checkpoint_path}: {exc}") from exc

        model.eval()
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._sheepdog_model_bundle = {
            "model": model,
            "tokenizer": tokenizer,
            "checkpoint_path": str(checkpoint_path),
        }

    def _predict_with_sheepdog(self, text: str, model_key: str) -> Dict[str, float | int | str]:
        self._ensure_sheepdog_model_loaded(model_key)
        assert torch is not None
        assert self._sheepdog_model_bundle is not None

        tokenizer = self._sheepdog_model_bundle["tokenizer"]
        model = self._sheepdog_model_bundle["model"]
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=self._sheepdog_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            _, binary_output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(binary_output, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred].item())

        return {
            "label": pred,
            "confidence": confidence,
            "model": model_key,
            "raw_response": f"checkpoint={self._sheepdog_model_bundle['checkpoint_path']}",
        }

    def _predict_with_loaded_model(self, text: str, model_key: str) -> Dict[str, float | int]:
        profile = MODEL_REGISTRY[model_key]
        cached = self._local_models.get(model_key)
        if cached is None:
            raise RuntimeError(f"Model '{model_key}' is not loaded")

        tokenizer = cached["tokenizer"]
        model = cached["model"]

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            max_length=profile["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred].item())

        return {
            "label": pred,
            "confidence": confidence,
        }

    def _load_few_shot_examples(self, k: int) -> List[Dict[str, str]]:
        if k <= 0:
            return []

        if k in self._few_shot_cache:
            return self._few_shot_cache[k]

        train_file = self._workspace_root / "GPT-Shot" / "clickbait_train.csv"
        if not train_file.exists():
            self._few_shot_cache[k] = []
            return []

        df = pd.read_csv(train_file, encoding="utf-8")
        if len(df) == 0:
            self._few_shot_cache[k] = []
            return []

        k_per_class = k // 2
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]

        positive = pos_df.sample(n=min(k_per_class, len(pos_df)), random_state=42)
        negative = neg_df.sample(n=min(k - k_per_class, len(neg_df)), random_state=42)

        examples: List[Dict[str, str]] = []
        for _, row in pd.concat([positive, negative]).iterrows():
            examples.append(
                {
                    "title": str(row["title"]),
                    "label": "Yes" if int(row["label"]) == 1 else "No",
                }
            )

        self._few_shot_cache[k] = examples
        return examples

    @staticmethod
    def _parse_yes_no_label(response: str) -> Tuple[int, bool]:
        normalized = re.sub(r"[^\w\s]", "", (response or "").lower().strip())
        if re.search(r"\byes\b", normalized):
            return 1, False
        if re.search(r"\bno\b", normalized):
            return 0, False
        return 0, True

    def _build_gpt_messages(self, text: str, few_shot_k: int) -> List[Dict[str, str]]:
        system_message = (
            "You are a clickbait detector. Determine if a title is clickbait.\n"
            "Answer ONLY with 'Yes' (if clickbait) or 'No' (if not clickbait).\n"
            "Do not provide any explanation or additional text."
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

        for example in self._load_few_shot_examples(few_shot_k):
            messages.append({"role": "user", "content": f"Title: {example['title']}"})
            messages.append({"role": "assistant", "content": example["label"]})

        messages.append({"role": "user", "content": f"Title: {text}"})
        return messages

    @staticmethod
    def _resolve_api_model_name(api_model: str, base_url: str) -> str:
        model_name = (api_model or "").strip()
        if not model_name:
            return model_name

        # Shopaikey proxy expects plain model names (no provider prefix).
        if "shopaikey.com" in (base_url or "") and "/" in model_name:
            return model_name.split("/", 1)[1]
        return model_name

    def _predict_with_gpt(
        self,
        text: str,
        model_key: str,
        api_key: str,
        custom_api_model: Optional[str] = None,
        custom_api_provider: Optional[str] = None,
        custom_api_base: Optional[str] = None,
    ) -> Dict[str, float | int | str]:
        if completion is None:
            raise RuntimeError("Missing dependency: install litellm to use GPT methods")

        profile = MODEL_REGISTRY[model_key]
        base_url = (custom_api_base or "").strip() or profile["base_url"]
        requested_provider = (custom_api_provider or "").strip()
        requested_api_model = (custom_api_model or "").strip() or profile["api_model"]
        api_model = self._resolve_api_model_name(requested_api_model, base_url=base_url)
        max_tokens = profile["max_tokens"]
        temperature = profile["temperature"]
        few_shot_k = int(profile.get("few_shot_k", 0))

        messages = self._build_gpt_messages(text, few_shot_k)
        prompt_preview = json.dumps(
            {
                "model": api_model,
                "base_url": base_url,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            ensure_ascii=False,
            indent=2,
        )

        request_kwargs: Dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_key": api_key,
            "api_base": base_url,
        }
        if "shopaikey.com" in (base_url or ""):
            request_kwargs["custom_llm_provider"] = "openai"
        elif requested_provider:
            request_kwargs["custom_llm_provider"] = requested_provider

        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                response = completion(**request_kwargs)
                break
            except Exception as e:
                error_msg = str(e).lower()
                if attempt < max_retries - 1 and ("rate limit" in error_msg or "high demand" in error_msg or "429" in error_msg or "503" in error_msg):
                    import time
                    time.sleep(2.0 * (attempt + 1))
                else:
                    raise

        raw_response = self._extract_litellm_text(response)
        label, ambiguous = self._parse_yes_no_label(raw_response)

        return {
            "label": label,
            "confidence": 1.0,
            "model": model_key,
            "api_model_used": api_model,
            "api_model_requested": requested_api_model,
            "raw_response": raw_response,
            "ambiguous": ambiguous,
            "prompt_preview": prompt_preview,
        }

    @staticmethod
    def _orcd_extract_quoted_text(text: str, input_text: str) -> str:
        old_text = text
        matches = re.findall(r"<(.*?)>", text)
        if matches:
            text = "".join(matches)

        cleaned_text = re.sub(r"['\"]", "", text)
        words_in_text = re.split(r"\s+", cleaned_text.strip())[:10]
        cleaned_input = re.sub(r"['\"]", "", input_text)
        words_in_input = re.split(r"\s+", cleaned_input.strip())[:10]

        if words_in_text == words_in_input:
            text = old_text

        for sentence in [
            "agree reasoning",
            "disagree reasoning",
            "Clickbait",
            "Non-clickbait",
            "increase",
            "lower",
        ]:
            text = re.sub(re.escape(sentence) + r"\s*", "", text)

        return text.strip()

    @staticmethod
    def _orcd_parse_int_score(raw: str) -> int:
        match = re.search(r"\d+", raw or "")
        if not match:
            return 50
        return max(0, min(100, int(match.group())))

    @staticmethod
    def _orcd_parse_score_output(raw: str) -> Optional[int]:
        text = (raw or "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "score" in parsed:
                value = int(parsed["score"])
                return max(0, min(100, value))
        except (JSONDecodeError, TypeError, ValueError):
            pass

        obj_match = re.search(r"\{[^{}]*\}", text)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
                if isinstance(parsed, dict) and "score" in parsed:
                    value = int(parsed["score"])
                    return max(0, min(100, value))
            except (JSONDecodeError, TypeError, ValueError):
                pass

        # Fallback for providers that ignore json schema hints.
        return ClickbaitModelService._orcd_parse_int_score(text)

    @staticmethod
    def _extract_litellm_text(response: Any) -> str:
        try:
            if isinstance(response, dict):
                choices = response.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    return str(message.get("content") or "").strip()
                return ""

            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                return str(getattr(message, "content", "") or "").strip()
            return ""
        except Exception:
            return ""

    def _orcd_chat_call(
        self,
        api_model: str,
        api_key: str,
        api_base: str,
        api_provider: Optional[str],
        prompt: str,
        max_tokens: int,
        temperature: float,
        timeout_s: float,
        enforce_json_score: bool = False,
    ) -> Dict[str, Any]:
        if completion is None:
            raise RuntimeError("Missing dependency: install litellm to use ORCD methods")

        use_response_format = bool(enforce_json_score)
        # Shopaikey proxy is OpenAI-compatible but can reject/slow response_format for score-only prompts.
        if "shopaikey.com" in (api_base or ""):
            use_response_format = False

        request_kwargs: Dict[str, Any] = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are a helpful and expert news analyst."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "api_key": api_key,
            "api_base": api_base,
            "timeout": timeout_s,
        }
        if "shopaikey.com" in (api_base or ""):
            request_kwargs["custom_llm_provider"] = "openai"
        elif api_provider:
            request_kwargs["custom_llm_provider"] = api_provider
        if use_response_format:
            request_kwargs["response_format"] = {"type": "json_object"}

        network_calls = 0
        fallback_used = False
        try:
            network_calls += 1
            response = completion(**request_kwargs)
        except Exception:
            if not use_response_format:
                raise
            request_kwargs.pop("response_format", None)
            fallback_used = True
            network_calls += 1
            response = completion(**request_kwargs)

        return {
            "text": self._extract_litellm_text(response),
            "network_calls": network_calls,
            "fallback_used": fallback_used,
            "response_format_used": use_response_format,
            "response_format_skipped": bool(enforce_json_score and not use_response_format),
        }

    def _run_orcd_reasoning_generation(
        self,
        text: str,
        model_key: str,
        api_key: str,
        custom_api_model: Optional[str] = None,
        custom_api_provider: Optional[str] = None,
        custom_api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        if completion is None:
            raise RuntimeError("Missing dependency: install litellm to use ORCD methods")

        profile = MODEL_REGISTRY[model_key]
        base_url = (custom_api_base or "").strip() or profile["base_url"]
        requested_provider = (custom_api_provider or "").strip()
        requested_api_model = (custom_api_model or "").strip() or profile["api_model"]
        api_model = self._resolve_api_model_name(requested_api_model, base_url=base_url)
        temperature = float(profile.get("temperature", 0.3))
        temp_score = float(profile.get("temperature_score", 0.0))
        temp_reasoning = float(profile.get("temperature_reasoning", temperature))
        max_tokens_score = int(profile.get("max_tokens_score", 16))
        max_tokens_reasoning = int(profile.get("max_tokens_reasoning", 150))
        fallback_retries = int(profile.get("max_retries", 3))
        max_retries_score = int(profile.get("max_retries_score", fallback_retries))
        max_retries_reasoning = int(profile.get("max_retries_reasoning", fallback_retries))
        max_agree_iterations = int(profile.get("max_agree_iterations", 20))
        max_disagree_iterations = int(profile.get("max_disagree_iterations", 20))
        disagree_analysis_every_n = max(1, int(profile.get("disagree_analysis_every_n", 3)))
        timeout_score_s = float(profile.get("timeout_score_s", 3.0))
        timeout_reasoning_s = float(profile.get("timeout_reasoning_s", 20.0))
        enable_payload_preview = bool(profile.get("enable_payload_preview", False))

        # Claude-family models can respond slower on multi-step ORCD prompts.
        # Increase timeout/retry budgets to reduce intermittent transport failures.
        model_hint = api_model.lower()
        if model_hint.startswith("gpt-5"):
            # GPT-5 family currently rejects non-default temperature values.
            temp_score = 1.0
            temp_reasoning = 1.0
        if "claude" in model_hint or "gemini" in model_hint:
            timeout_score_s = max(timeout_score_s, 10.0)
            timeout_reasoning_s = max(timeout_reasoning_s, 45.0)
            max_retries_score = max(max_retries_score, 2)
            max_retries_reasoning = max(max_retries_reasoning, 4)

        prompt1 = "Goal: As a news expert, score the title's content to determine its accuracy and completeness, and assess people's agreement with the title.\n"
        prompt1 += "Requirement 1: The title content is {}.\n"
        prompt1 += "Requirement 2: The score range is from 0 to 100, where 0 means complete disagreement, 50 means difficult to judge, and 100 means complete agreement. The score should be humanized and not restricted to multiples of 5.\n"
        prompt1 += "Requirement 3: Output format[int].\n"

        prompt_reassign = "Goal: Re-assess the agreement level based on the title's content."
        prompt_reassign += "Requirement 1: The content of the title is {}.\n"
        prompt_reassign += "Requirement 2: Consider the previous agreement score for the title, which was {}.\n"
        prompt_reassign += "Requirement 3: The new score should fall within the range of {} to {}.\n"
        prompt_reassign += "Requirement 4: The score should be between 0 and 100 and not restricted to multiples of 5.\n"
        prompt_reassign += "Requirement 5: The output format is [int]."

        prompt2 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people believe the content in the title.\n"
        prompt2 += "Requirement 1: The title content is {}.\n"
        prompt2 += "Requirement 2: Please agree with the title content in combination with the following four aspects:\n"
        prompt2 += "1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n"
        prompt2 += "2. Logic: Are there any leaps in reasoning or inconsistencies?\n"
        prompt2 += "3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n"
        prompt2 += "4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n"
        prompt2 += "Requirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\n"
        prompt2 += "Requirement 4: The output format is [reasoning content].\n"

        prompt3 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people disbelieve the content in the title.\n"
        prompt3 += "Requirement 1: The title content is {}.\n"
        prompt3 += "Requirement 2: Please disagree with the title content in combination with the following four aspects:\n"
        prompt3 += "1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n"
        prompt3 += "2. Logic: Are there any leaps in reasoning or inconsistencies?\n"
        prompt3 += "3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n"
        prompt3 += "4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n"
        prompt3 += "Requirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\n"
        prompt3 += "Requirement 4: The output format is [reasoning content].\n"

        prompt4 = "Goal:  Re-score based on the title content, initial score, and {}reasoning."
        prompt4 += "Requirement 1: The title is {}.\n"
        prompt4 += "Requirement 2: The initial score is {}.\n"
        prompt4 += "Requirement 3: The {} reasoning content is {}.\n"
        prompt4 += "Requirement 4: The score should be between 0 and 100 and not restricted to multiples of 5.\n"
        prompt4 += "Requirement 5: The output format is [int].\n"

        prompt5 = "Goal: Analyze the {} reasoning content from the perspectives of rationality and logic.\n"
        prompt5 += "Requirement 1: Consider the previous {} reasoning content: {}\n"
        prompt5 += "Requirement 2: Consider the previous score based on the {} reasoning: {}.\n"
        prompt5 += "Requirement 3: The analysis should be limited to 50-70 words.\n"
        prompt5 += "Requirement 4: Output format [reasoning content].\n"

        prompt6 = "Goal: Regenerate {} reasoning content, because the previous reasoning did not effectively {} the title's recognition score\n"
        prompt6 += "Requirement 1: The title is {}.\n"
        prompt6 += "Requirement 2: The initial score is {}.\n"
        prompt6 += "Requirement 3: Consider the previous {} reasoning content: {}.\n"
        prompt6 += "Requirement 4: Consider the title score based on the previous {} reasoning: {}.\n"
        prompt6 += "Requirement 5: Consider the evaluation of the reasoning for {}: {}. \n"
        prompt6 += "Requirement 6: Analyze the logical inconsistencies in the previous reasoning and explain why the new reasoning is more suitable for the title content.\n"
        prompt6 += "Requirement 7: New inference generation should combine the following four aspects and adapt to the content of the title to {} people's identification with the content of the title and make people {} in the content of the title."
        prompt6 += "1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n"
        prompt6 += "2. Logic: Are there any leaps in reasoning or inconsistencies?\n"
        prompt6 += "3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n"
        prompt6 += "4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n"
        prompt6 += "Requirement 8: The limit for inference is 40-60 words, and the limit for explanation is 20-40 words. The inference content is placed in [] and the explanation content is placed in ().\n"
        prompt6 += "Requirement 9: Output format is [Reasoning Content] (Explanatory Content).\n"
        prompt6 += "Requirement 10: The score should still range from 0 to 100, and it should be more humanized, not restricted to multiples of 5.\n"
        prompt6 += "Requirement 11: Output format for the score is [int].\n"

        str3 = "agree"
        str4 = "disagree"
        str5 = "increase"
        str6 = "lower"
        str7 = "believe"
        str8 = "disbelieve"

        process_log: List[Dict[str, Any]] = []
        payload_preview: List[Dict[str, Any]] = []
        original_attempts: List[Dict[str, Any]] = []
        telemetry: Dict[str, Any] = {
            "api_calls": 0,
            "api_network_calls": 0,
            "transport_fallbacks": 0,
            "score_response_format_skipped": 0,
            "api_time_ms": 0.0,
            "gates": {},
        }
        api_blocker: Optional[str] = None

        def _is_non_retryable_api_error(error_text: str) -> bool:
            normalized = (error_text or "").lower()
            non_retryable_markers = (
                "unsupportedparamserror",
                "does not support parameters",
                "invalid token",
                "invalid api key",
                "incorrect api key",
                "unauthorized",
                "authentication",
                "permission",
                "insufficient_quota",
                "rate limit",
            )
            return any(marker in normalized for marker in non_retryable_markers)

        def _compact_error(error_text: str) -> str:
            one_line = " ".join((error_text or "").split())
            return one_line[:280]

        def call_step(
            step: str,
            prompt_text: str,
            is_score: bool,
            temp: float,
            max_tokens: int,
            gate: str,
        ) -> str:
            if enable_payload_preview:
                payload_preview.append(
                    {
                        "step": step,
                        "request": {
                            "model": api_model,
                            "base_url": base_url,
                            "messages": [
                                {"role": "system", "content": "You are a helpful and expert news analyst."},
                                {"role": "user", "content": prompt_text},
                            ],
                            "max_tokens": max_tokens,
                            "temperature": temp,
                        },
                    }
                )

            started = time.perf_counter()
            call_result = self._orcd_chat_call(
                api_model=api_model,
                api_key=api_key,
                api_base=base_url,
                api_provider=requested_provider,
                prompt=prompt_text,
                max_tokens=max_tokens,
                temperature=temp,
                timeout_s=(timeout_score_s if is_score else timeout_reasoning_s),
                enforce_json_score=is_score,
            )
            raw = str(call_result.get("text", "") or "")
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            telemetry["api_calls"] += 1
            telemetry["api_network_calls"] += int(call_result.get("network_calls", 1))
            if call_result.get("fallback_used"):
                telemetry["transport_fallbacks"] += 1
            if call_result.get("response_format_skipped"):
                telemetry["score_response_format_skipped"] += 1
            telemetry["api_time_ms"] += duration_ms
            process_log.append(
                {
                    "step": step,
                    "kind": "score" if is_score else "reasoning",
                    "raw": raw if enable_payload_preview else "",
                    "duration_ms": duration_ms,
                    "gate": gate,
                }
            )
            return raw

        def generate_score(prompt_text: str, step: str, gate: str) -> int:
            nonlocal api_blocker
            if api_blocker is not None:
                return 50
            raw = ""
            for attempt in range(max_retries_score):
                try:
                    raw = call_step(
                        step=f"{step}_attempt_{attempt + 1}",
                        prompt_text=prompt_text,
                        is_score=True,
                        temp=temp_score,
                        max_tokens=max_tokens_score,
                        gate=gate,
                    )
                except Exception as exc:  # noqa: BLE001
                    error_text = str(exc)
                    process_log.append(
                        {
                            "step": f"{step}_attempt_{attempt + 1}",
                            "kind": "score",
                            "error": error_text,
                            "gate": gate,
                        }
                    )
                    if _is_non_retryable_api_error(error_text):
                        api_blocker = _compact_error(error_text)
                        break
                    if attempt < max_retries_score - 1:
                        time.sleep(min(2.0, 0.5 * (attempt + 1)))
                    continue

                parsed = self._orcd_parse_score_output(raw)
                if parsed is not None:
                    return parsed
            
            # Instead of setting a fatal global blocker for a transient LLM failure,
            # just return 50 to allow the pipeline gate loop to trigger a retry.
            return 50

        def generate_res(prompt_text: str, step: str, gate: str) -> str:
            nonlocal api_blocker
            if api_blocker is not None:
                return f"Error: {api_blocker}"
            raw = ""
            for attempt in range(max_retries_reasoning):
                try:
                    raw = call_step(
                        step=f"{step}_attempt_{attempt + 1}",
                        prompt_text=prompt_text,
                        is_score=False,
                        temp=temp_reasoning,
                        max_tokens=max_tokens_reasoning,
                        gate=gate,
                    )
                except Exception as exc:  # noqa: BLE001
                    error_text = str(exc)
                    process_log.append(
                        {
                            "step": f"{step}_attempt_{attempt + 1}",
                            "kind": "reasoning",
                            "error": error_text,
                            "gate": gate,
                        }
                    )
                    if _is_non_retryable_api_error(error_text):
                        api_blocker = _compact_error(error_text)
                        break
                    if attempt < max_retries_reasoning - 1:
                        time.sleep(min(2.0, 0.5 * (attempt + 1)))
                    continue

                if raw and raw.strip():
                    return raw.strip()
            
            if api_blocker is None:
                api_blocker = "Failed to generate reasoning response after retries"
            return f"Error: {api_blocker}"

        original_pass_condition = lambda s: 30 <= s <= 70

        seed_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as seed_executor:
            original_gate_started = time.perf_counter()
            original_score_future = seed_executor.submit(generate_score, prompt1.format(text), "original_score", "original")
            agree_reason_future = seed_executor.submit(generate_res, prompt2.format(text), "agree_reason", "agree")
            disagree_reason_future = seed_executor.submit(generate_res, prompt3.format(text), "disagree_reason", "disagree")

            original_score = int(original_score_future.result())
            original_attempts.append(
                {
                    "attempt": len(original_attempts) + 1,
                    "source": "initial",
                    "original_score": original_score,
                    "original_socre": original_score,
                    "passed": original_pass_condition(original_score),
                }
            )
            count0 = 0
            while original_score < 30 or original_score > 70:
                if api_blocker is not None:
                    break
                original_score = generate_score(
                    prompt_reassign.format(text, original_score, 30, 70),
                    step="original_score_reassign",
                    gate="original",
                )
                original_attempts.append(
                    {
                        "attempt": len(original_attempts) + 1,
                        "source": "reassign",
                        "original_score": original_score,
                        "original_socre": original_score,
                        "passed": original_pass_condition(original_score),
                    }
                )
                count0 += 1
                if count0 >= 3:
                    break
            telemetry["gates"]["original_gate_ms"] = round((time.perf_counter() - original_gate_started) * 1000, 2)

            initial_agree_reason_raw = agree_reason_future.result()
            initial_disagree_reason_raw = disagree_reason_future.result()

        telemetry["gates"]["seed_parallel_ms"] = round((time.perf_counter() - seed_started) * 1000, 2)
        initial_agree_reason = self._orcd_extract_quoted_text(initial_agree_reason_raw, text)
        initial_disagree_reason = self._orcd_extract_quoted_text(initial_disagree_reason_raw, text)

        def run_agree_gate() -> Dict[str, Any]:
            gate_started = time.perf_counter()
            agree_iterations: List[Dict[str, Any]] = []
            agree_analysis_history: List[str] = []

            agree_reason = initial_agree_reason

            agr_score = generate_score(
                prompt4.format(str3, text, original_score, str3, agree_reason),
                step="agree_score",
                gate="agree",
            )
            agree_pass_condition = lambda s: (s - original_score >= 10) and (s > 55)
            agree_iterations.append(
                {
                    "iteration": len(agree_iterations) + 1,
                    "source": "initial",
                    "agree_score": agr_score,
                    "agree_reason": agree_reason,
                    "aggre_reason": agree_reason,
                    "passed": agree_pass_condition(agr_score),
                }
            )

            count1 = 0
            while agr_score - original_score < 10 or agr_score <= 55:
                if api_blocker is not None:
                    break
                prev_agree_reason_for_prompt = [agree_reason]
                ret_agree_reason = generate_res(
                    prompt5.format(str3, str3, prev_agree_reason_for_prompt, str3, agr_score),
                    step="agree_reason_analysis",
                    gate="agree",
                )
                agree_analysis_history.append(ret_agree_reason)

                agree_reason = generate_res(
                    prompt6.format(
                        str3,
                        str5,
                        text,
                        original_score,
                        str3,
                        prev_agree_reason_for_prompt,
                        str3,
                        agr_score,
                        str3,
                        ret_agree_reason,
                        str5,
                        str7,
                    ),
                    step="agree_reason_regenerate",
                    gate="agree",
                )
                agree_reason = self._orcd_extract_quoted_text(agree_reason, text)

                agr_score = generate_score(
                    prompt4.format(str3, text, original_score, str3, agree_reason),
                    step="agree_score_regenerate",
                    gate="agree",
                )
                agree_iterations.append(
                    {
                        "iteration": len(agree_iterations) + 1,
                        "source": "regenerated",
                        "agree_score": agr_score,
                        "agree_reason": agree_reason,
                        "aggre_reason": agree_reason,
                        "passed": agree_pass_condition(agr_score),
                    }
                )

                count1 += 1
                if count1 == max_agree_iterations:
                    break

            return {
                "iterations": agree_iterations,
                "analysis_history": agree_analysis_history,
                "final_reason": agree_iterations[-1]["agree_reason"] if agree_iterations else "",
                "final_score": agr_score,
                "passed": agree_pass_condition(agr_score),
                "gate_ms": round((time.perf_counter() - gate_started) * 1000, 2),
            }

        def run_disagree_gate() -> Dict[str, Any]:
            gate_started = time.perf_counter()
            disagree_iterations: List[Dict[str, Any]] = []
            disagree_analysis_history: List[str] = []

            disagree_reason = initial_disagree_reason

            dis_score = generate_score(
                prompt4.format(str4, text, original_score, str4, disagree_reason),
                step="disagree_score",
                gate="disagree",
            )
            disagree_pass_condition = lambda s: (original_score - s >= 10) and (s < 45)
            disagree_iterations.append(
                {
                    "iteration": len(disagree_iterations) + 1,
                    "source": "initial",
                    "dis_score": dis_score,
                    "dis_reason": disagree_reason,
                    "passed": disagree_pass_condition(dis_score),
                }
            )

            count2 = 0
            prev_dis_score = dis_score
            no_improvement_runs = 0
            while original_score - dis_score < 10 or dis_score >= 45:
                if api_blocker is not None:
                    break
                prev_disagree_reason_for_prompt = [disagree_reason]
                # Analysis is useful but expensive; run it periodically to reduce API pressure.
                if count2 % disagree_analysis_every_n == 0:
                    ret_disagree_reason = generate_res(
                        prompt5.format(str4, str4, prev_disagree_reason_for_prompt, str4, dis_score),
                        step="disagree_reason_analysis",
                        gate="disagree",
                    )
                else:
                    ret_disagree_reason = "Strengthen contradiction with the title and lower credibility further."
                disagree_analysis_history.append(ret_disagree_reason)

                disagree_reason = generate_res(
                    prompt6.format(
                        str4,
                        str6,
                        text,
                        original_score,
                        str4,
                        prev_disagree_reason_for_prompt,
                        str4,
                        dis_score,
                        str4,
                        ret_disagree_reason,
                        str6,
                        str8,
                    ),
                    step="disagree_reason_regenerate",
                    gate="disagree",
                )
                disagree_reason = self._orcd_extract_quoted_text(disagree_reason, text)

                dis_score = generate_score(
                    prompt4.format(str4, text, original_score, str4, disagree_reason),
                    step="disagree_score_regenerate",
                    gate="disagree",
                )
                disagree_iterations.append(
                    {
                        "iteration": len(disagree_iterations) + 1,
                        "source": "regenerated",
                        "dis_score": dis_score,
                        "dis_reason": disagree_reason,
                        "passed": disagree_pass_condition(dis_score),
                    }
                )

                if dis_score < prev_dis_score:
                    no_improvement_runs = 0
                else:
                    no_improvement_runs += 1
                prev_dis_score = dis_score

                # If the score stalls, do a constrained re-score call to avoid long retry tails.
                if no_improvement_runs >= 2 and (original_score - dis_score < 10 or dis_score >= 45):
                    dis_score = generate_score(
                        prompt_reassign.format(text, dis_score, 0, 44),
                        step="disagree_score_reassign",
                        gate="disagree",
                    )
                    disagree_iterations.append(
                        {
                            "iteration": len(disagree_iterations) + 1,
                            "source": "reassign",
                            "dis_score": dis_score,
                            "dis_reason": disagree_reason,
                            "passed": disagree_pass_condition(dis_score),
                        }
                    )
                    no_improvement_runs = 0

                count2 += 1
                if count2 == max_disagree_iterations:
                    break

            return {
                "iterations": disagree_iterations,
                "analysis_history": disagree_analysis_history,
                "final_reason": disagree_iterations[-1]["dis_reason"] if disagree_iterations else "",
                "final_score": dis_score,
                "passed": disagree_pass_condition(dis_score),
                "gate_ms": round((time.perf_counter() - gate_started) * 1000, 2),
            }

        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            agree_future = executor.submit(run_agree_gate)
            disagree_future = executor.submit(run_disagree_gate)
            agree_result = agree_future.result()
            disagree_result = disagree_future.result()
        telemetry["gates"]["parallel_branches_ms"] = round((time.perf_counter() - parallel_started) * 1000, 2)
        agree_iterations = agree_result["iterations"]
        disagree_iterations = disagree_result["iterations"]
        agree_analysis_history = agree_result["analysis_history"]
        disagree_analysis_history = disagree_result["analysis_history"]
        final_agree_reason = agree_result["final_reason"]
        final_disagree_reason = disagree_result["final_reason"]
        agr_score = int(agree_result["final_score"])
        dis_score = int(disagree_result["final_score"])
        agree_passed = bool(agree_result["passed"])
        disagree_passed = bool(disagree_result["passed"])
        telemetry["gates"]["agree_gate_ms"] = agree_result["gate_ms"]
        telemetry["gates"]["disagree_gate_ms"] = disagree_result["gate_ms"]

        agree_reason_all = "$$$$$ ".join(item["agree_reason"] for item in agree_iterations)
        disagree_reason_all = "$$$$$ ".join(item["dis_reason"] for item in disagree_iterations)
        original_score_all = "$$ ".join(str(item["original_score"]) for item in original_attempts)
        agree_score_all = "$$ ".join(str(item["agree_score"]) for item in agree_iterations)
        disagree_score_all = "$$ ".join(str(item["dis_score"]) for item in disagree_iterations)

        flow_trace = {
            "original_gate": {
                "condition": "30 <= original_score <= 70",
                "attempts": original_attempts,
                "final_score": original_score,
                "final_socre": original_score,
                "passed": original_pass_condition(original_score),
            },
            "agree_gate": {
                "condition": "agree_score - original_score >= 10 and agree_score > 55",
                "iterations": agree_iterations,
                "analysis_history": agree_analysis_history,
                "final_score": agr_score,
                "final_reason": final_agree_reason,
                "final_aggre_reason": final_agree_reason,
                "passed": agree_passed,
            },
            "disagree_gate": {
                "condition": "original_score - dis_score >= 10 and dis_score < 45",
                "iterations": disagree_iterations,
                "analysis_history": disagree_analysis_history,
                "final_score": dis_score,
                "final_reason": final_disagree_reason,
                "passed": disagree_passed,
            },
        }

        return {
            "title": text,
            "api_model_used": api_model,
            "api_model_requested": requested_api_model,
            "agree_reason": final_agree_reason,
            "aggre_reason": final_agree_reason,
            "disagree_reason": final_disagree_reason,
            "original_score": original_score,
            "original_socre": original_score,
            "agree_score": agr_score,
            "disagree_score": dis_score,
            "process_log": process_log,
            "payload_preview": json.dumps(payload_preview, ensure_ascii=False, indent=2) if enable_payload_preview else "",
            "flow_trace": flow_trace,
            "loop_history": [],
            "original_score_all": original_score_all,
            "agree_score_all": agree_score_all,
            "disagree_score_all": disagree_score_all,
            "agree_reason_all": agree_reason_all,
            "disagree_reason_all": disagree_reason_all,
            "agree_reason_reviews": agree_analysis_history,
            "disagree_reason_reviews": disagree_analysis_history,
            "telemetry": {
                "api_calls": telemetry["api_calls"],
                "api_network_calls": telemetry["api_network_calls"],
                "transport_fallbacks": telemetry["transport_fallbacks"],
                "score_response_format_skipped": telemetry["score_response_format_skipped"],
                "api_failure_reason": api_blocker or "",
                "api_time_ms": round(float(telemetry["api_time_ms"]), 2),
                "gates": telemetry["gates"],
            },
        }

    def _load_orcd_modelbart_module(self):
        modelbart_path = self._orcd_base_dir / "train" / "modelbart.py"
        if not modelbart_path.exists():
            raise FileNotFoundError(f"Missing ORCD model definition: {modelbart_path}")

        module_name = "gui_orcd_modelbart"
        spec = importlib.util.spec_from_file_location(module_name, str(modelbart_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load ORCD model module")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _ensure_orcd_model_loaded(self) -> None:
        if self._orcd_model_bundle is not None and self._orcd_tokenizer is not None:
            return
        if torch is None or BertTokenizer is None:
            raise RuntimeError("Missing dependencies: install torch and transformers to run ORCD inference")
        if not self._orcd_weight_path.exists():
            raise FileNotFoundError(f"Missing ORCD weight file: {self._orcd_weight_path}")

        module = self._load_orcd_modelbart_module()
        device = self._device

        bert = module.BertEncoder(256, False).to(device)
        bert2 = module.BertEncoder(256, False).to(device)
        bert3 = module.BertEncoder(256, False).to(device)
        attention = module.Attention_Encoder().to(device)
        r2t_usefulness = module.Similarity().to(device)
        t2r_usefulness = module.Similarity().to(device)
        reason_usefulness = module.Reason_Similarity().to(device)
        aggregator = module.Aggregator().to(device)
        detection_module = module.DetectionModule().to(device)

        checkpoint = torch.load(str(self._orcd_weight_path), map_location=device)

        if isinstance(checkpoint, dict):
            if "bert" in checkpoint:
                bert.load_state_dict(checkpoint["bert"])
                bert2.load_state_dict(checkpoint["bert2"])
                bert3.load_state_dict(checkpoint["bert3"])
                attention.load_state_dict(checkpoint["attention"])
                r2t_usefulness.load_state_dict(checkpoint["R2T_usefulness"])
                t2r_usefulness.load_state_dict(checkpoint["T2R_usefulness"])
                reason_usefulness.load_state_dict(checkpoint["Reason_usefulness"])
                aggregator.load_state_dict(checkpoint["aggregator"])
                detection_module.load_state_dict(checkpoint["detection_module"])
            elif any(str(k).startswith("bert.") for k in checkpoint.keys()):
                bert_dict = {
                    k.replace("bert.", ""): v
                    for k, v in checkpoint.items()
                    if str(k).startswith("bert.") and not str(k).startswith("bert2.") and not str(k).startswith("bert3.")
                }
                bert2_dict = {k.replace("bert2.", ""): v for k, v in checkpoint.items() if str(k).startswith("bert2.")}
                bert3_dict = {k.replace("bert3.", ""): v for k, v in checkpoint.items() if str(k).startswith("bert3.")}
                attention_dict = {k.replace("attention.", ""): v for k, v in checkpoint.items() if str(k).startswith("attention.")}
                r2t_dict = {
                    k.replace("R2T_usefulness.", ""): v
                    for k, v in checkpoint.items()
                    if str(k).startswith("R2T_usefulness.")
                }
                t2r_dict = {
                    k.replace("T2R_usefulness.", ""): v
                    for k, v in checkpoint.items()
                    if str(k).startswith("T2R_usefulness.")
                }
                reason_dict = {
                    k.replace("Reason_usefulness.", ""): v
                    for k, v in checkpoint.items()
                    if str(k).startswith("Reason_usefulness.")
                }
                agg_dict = {k.replace("aggregator.", ""): v for k, v in checkpoint.items() if str(k).startswith("aggregator.")}
                det_dict = {
                    k.replace("detection_module.", ""): v
                    for k, v in checkpoint.items()
                    if str(k).startswith("detection_module.")
                }

                if bert_dict:
                    bert.load_state_dict(bert_dict, strict=False)
                if bert2_dict:
                    bert2.load_state_dict(bert2_dict, strict=False)
                if bert3_dict:
                    bert3.load_state_dict(bert3_dict, strict=False)
                if attention_dict:
                    attention.load_state_dict(attention_dict, strict=False)
                if r2t_dict:
                    r2t_usefulness.load_state_dict(r2t_dict, strict=False)
                if t2r_dict:
                    t2r_usefulness.load_state_dict(t2r_dict, strict=False)
                if reason_dict:
                    reason_usefulness.load_state_dict(reason_dict, strict=False)
                if agg_dict:
                    aggregator.load_state_dict(agg_dict, strict=False)
                if det_dict:
                    detection_module.load_state_dict(det_dict, strict=False)

        for model_obj in [
            bert,
            bert2,
            bert3,
            attention,
            r2t_usefulness,
            t2r_usefulness,
            reason_usefulness,
            aggregator,
            detection_module,
        ]:
            model_obj.eval()

        self._orcd_model_bundle = {
            "bert": bert,
            "bert2": bert2,
            "bert3": bert3,
            "attention": attention,
            "r2t_usefulness": r2t_usefulness,
            "t2r_usefulness": t2r_usefulness,
            "reason_usefulness": reason_usefulness,
            "aggregator": aggregator,
            "detection_module": detection_module,
        }
        self._orcd_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _orcd_tokenize(self, text: str):
        assert self._orcd_tokenizer is not None
        output = self._orcd_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._orcd_max_len,
        )
        return output["input_ids"]

    def _predict_with_orcd(
        self,
        text: str,
        model_key: str,
        api_key: str,
        custom_api_model: Optional[str] = None,
        custom_api_provider: Optional[str] = None,
        custom_api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        load_started = time.perf_counter()
        self._ensure_orcd_model_loaded()
        load_model_ms = round((time.perf_counter() - load_started) * 1000, 2)

        generation_started = time.perf_counter()
        generated = self._run_orcd_reasoning_generation(
            text=text,
            model_key=model_key,
            api_key=api_key,
            custom_api_model=custom_api_model,
            custom_api_provider=custom_api_provider,
            custom_api_base=custom_api_base,
        )
        reasoning_generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
        assert torch is not None
        assert self._orcd_model_bundle is not None

        device = self._device
        content_ids = torch.tensor([self._orcd_tokenize(generated["title"])], device=device).long()
        pos_ids = torch.tensor([self._orcd_tokenize(generated["agree_reason"])], device=device).long()
        neg_ids = torch.tensor([self._orcd_tokenize(generated["disagree_reason"])], device=device).long()

        bert = self._orcd_model_bundle["bert"]
        bert2 = self._orcd_model_bundle["bert2"]
        bert3 = self._orcd_model_bundle["bert3"]
        attention = self._orcd_model_bundle["attention"]
        r2t_usefulness = self._orcd_model_bundle["r2t_usefulness"]
        t2r_usefulness = self._orcd_model_bundle["t2r_usefulness"]
        reason_usefulness = self._orcd_model_bundle["reason_usefulness"]
        aggregator = self._orcd_model_bundle["aggregator"]
        detection_module = self._orcd_model_bundle["detection_module"]

        inference_started = time.perf_counter()
        with torch.no_grad():
            content = bert(content_ids)
            positive = bert2(pos_ids)
            negative = bert3(neg_ids)

            pos_reason2text, pos_text2reason, positive, neg_reason2text, neg_text2reason, negative = attention(
                content,
                positive,
                negative,
            )

            _, r2t_aligned_agr, _ = r2t_usefulness(content, pos_reason2text)
            _, t2r_aligned_agr, _ = t2r_usefulness(content, pos_text2reason)
            _, r_aligned_agr, _ = reason_usefulness(content, positive)
            _, r2t_aligned_dis, _ = r2t_usefulness(content, neg_reason2text)
            _, t2r_aligned_dis, _ = t2r_usefulness(content, neg_text2reason)
            _, r_aligned_dis, _ = reason_usefulness(content, negative)

            final_feature = aggregator(
                content,
                r2t_aligned_agr,
                t2r_aligned_agr,
                r_aligned_agr,
                r2t_aligned_dis,
                t2r_aligned_dis,
                r_aligned_dis,
            )
            logits = detection_module(final_feature)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred].item())
        model_inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)

        telemetry = dict(generated.get("telemetry", {}))
        telemetry["model_load_ms"] = load_model_ms
        telemetry["reasoning_generation_ms"] = reasoning_generation_ms
        telemetry["model_inference_ms"] = model_inference_ms

        return {
            "label": pred,
            "confidence": confidence,
            "model": model_key,
            "api_model_used": generated.get("api_model_used", ""),
            "raw_response": json.dumps(
                {
                    "original_score": generated["original_score"],
                    "agree_score": generated["agree_score"],
                    "disagree_score": generated["disagree_score"],
                },
                ensure_ascii=False,
            ),
            "prompt_preview": generated["payload_preview"],
            "orcd_process": generated["process_log"],
            "telemetry": telemetry,
            "orcd_generated": {
                "agree_reason": generated["agree_reason"],
                "aggre_reason": generated.get("aggre_reason", generated["agree_reason"]),
                "disagree_reason": generated["disagree_reason"],
                "original_score": generated["original_score"],
                "original_socre": generated.get("original_socre", generated["original_score"]),
                "agree_score": generated["agree_score"],
                "disagree_score": generated["disagree_score"],
                "flow_trace": generated.get("flow_trace", {}),
                "loop_history": generated.get("loop_history", []),
                "original_score_all": generated.get("original_score_all", ""),
                "agree_score_all": generated.get("agree_score_all", ""),
                "disagree_score_all": generated.get("disagree_score_all", ""),
                "agree_reason_all": generated.get("agree_reason_all", ""),
                "disagree_reason_all": generated.get("disagree_reason_all", ""),
            },
        }

    def predict(
        self,
        text: str,
        model_key: str,
        api_key: Optional[str] = None,
        custom_api_model: Optional[str] = None,
        custom_api_provider: Optional[str] = None,
        custom_api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("Text must not be empty")
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {model_key}")

        profile = MODEL_REGISTRY[model_key]
        backend = profile.get("backend")

        if backend == "gpt":
            clean_api_key = (api_key or "").strip()
            if not clean_api_key:
                raise ValueError("API key is required for API-based models")
            result = self._predict_with_gpt(
                clean_text,
                model_key,
                clean_api_key,
                custom_api_model=(custom_api_model or "").strip(),
                custom_api_provider=(custom_api_provider or "").strip(),
                custom_api_base=(custom_api_base or "").strip(),
            )
            result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            return result

        if backend == "orcd":
            clean_api_key = (api_key or "").strip()
            if not clean_api_key:
                raise ValueError("API key is required for ORCD methods")
            result = self._predict_with_orcd(
                clean_text,
                model_key,
                clean_api_key,
                custom_api_model=(custom_api_model or "").strip(),
                custom_api_provider=(custom_api_provider or "").strip(),
                custom_api_base=(custom_api_base or "").strip(),
            )
            result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            return result

        if backend == "sheepdog":
            self.load_model(model_key)
            try:
                result = self._predict_with_sheepdog(clean_text, model_key)
            except RuntimeError as exc:
                if torch is not None and getattr(self._device, "type", "cpu") == "cuda" and "out of memory" in str(exc).lower():
                    self._device = torch.device("cpu")
                    self._clear_sheepdog_cache()
                    result = self._predict_with_sheepdog(clean_text, model_key)
                else:
                    raise
            result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            return result

        if backend == "generate_and_predict":
            clean_api_key = (api_key or "").strip()
            if not clean_api_key:
                raise ValueError("API key is required for generate-and-predict")
            result = self._predict_with_generate_and_predict(
                clean_text,
                model_key,
                clean_api_key,
                custom_api_model=(custom_api_model or "").strip(),
                custom_api_provider=(custom_api_provider or "").strip(),
                custom_api_base=(custom_api_base or "").strip(),
            )
            result["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
            return result

        self.load_model(model_key)

        try:
            result = self._predict_with_loaded_model(clean_text, model_key)
        except RuntimeError as exc:
            # Graceful fallback when a low-VRAM device triggers CUDA OOM.
            if torch is not None and getattr(self._device, "type", "cpu") == "cuda" and "out of memory" in str(exc).lower():
                self._device = torch.device("cpu")
                self._clear_local_cache()
                self.preload_all_local_models()
                result = self._predict_with_loaded_model(clean_text, model_key)
            else:
                raise

        result_payload: Dict[str, Any] = dict(result)
        result_payload["model"] = model_key
        result_payload["elapsed_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return result_payload

    # ------------------------------------------------------------------
    # generate_and_predict pipeline: GPT-3.5 scoring + ModelBART inference
    # ------------------------------------------------------------------

    def _build_generate_and_predict_reason(self, score: int, is_clickbait: bool) -> str:
        if is_clickbait:
            base = (
                "The title presents an engaging topic that aligns with common reader interests. "
                "Logically, it invites curiosity without overreaching. "
                "The content appears complete and the language remains relatively neutral, "
                f"supporting a moderate belief level of {score}/100."
            )
        else:
            base = (
                "The title presents factual information that aligns with established knowledge. "
                "Logically, it avoids sensationalism and follows a straightforward narrative. "
                "The content is complete and the language is objective, "
                f"supporting a moderate belief level of {score}/100."
            )
        return f'["[{base}]"]'

    def _run_gpt35_score_generation(
        self,
        title: str,
        api_key: str,
        api_model: str,
        api_base: str,
        api_provider: str,
    ) -> Dict[str, int]:
        prompt = (
            "Goal: As a news expert, evaluate the title's content and score it according to the criteria below.\n"
            "Requirement 1: The title is '" + title + "'.\n"
            "Requirement 2: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity.\n"
            "Requirement 3: First, assign an \"original_score\" representing the general public's agreement/belief level with the title (30 to 70).\n"
            "Requirement 4: Then, formulate an \"agree_reason\" (40-60 words) that advocates for the title being completely truthful. Based on this, assign an \"agree_score\" that must be at least 15 points higher than original_score (up to 100).\n"
            "Requirement 5: Finally, formulate a \"disagree_reason\" (40-60 words) that highlights any illogical leaps or vague language. Based on this, assign a \"disagree_score\" that must be at least 15 points lower than original_score (down to 0).\n"
            "Requirement 6: All scores should be strictly single integers.\n"
            "Requirement 7: The output MUST be a valid JSON object with EXACTLY three numeric fields: \"original_score\", \"agree_score\", \"disagree_score\". Do not output the reason text, just the final scores."
        )

        request_kwargs: Dict[str, Any] = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You are a helpful and expert news analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "api_key": api_key,
            "api_base": api_base,
            "timeout": 30.0,
        }
        if "shopaikey.com" in (api_base or ""):
            request_kwargs["custom_llm_provider"] = "openai"
        elif api_provider:
            request_kwargs["custom_llm_provider"] = api_provider

        if completion is None:
            raise RuntimeError("Missing dependency: install litellm to use generate-and-predict")

        response = completion(**request_kwargs)
        raw = self._extract_litellm_text(response)
        try:
            data = json.loads(raw)
            return {
                "original_score": int(data.get("original_score", 50)),
                "agree_score": int(data.get("agree_score", 80)),
                "disagree_score": int(data.get("disagree_score", 20)),
            }
        except Exception:
            return {"original_score": 50, "agree_score": 85, "disagree_score": 35}

    def _tokenize_gnp(self, text: str) -> List[int]:
        assert self._orcd_tokenizer is not None
        output = self._orcd_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self._orcd_max_len,
        )
        return output["input_ids"]

    def _predict_with_generate_and_predict(
        self,
        text: str,
        model_key: str,
        api_key: str,
        custom_api_model: Optional[str] = None,
        custom_api_provider: Optional[str] = None,
        custom_api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        profile = MODEL_REGISTRY[model_key]
        base_url = (custom_api_base or "").strip() or profile["base_url"]
        requested_provider = (custom_api_provider or "").strip()
        requested_api_model = (custom_api_model or "").strip() or profile["api_model"]
        api_model = self._resolve_api_model_name(requested_api_model, base_url=base_url)

        generation_started = time.perf_counter()
        scores = self._run_gpt35_score_generation(
            title=text,
            api_key=api_key,
            api_model=api_model,
            api_base=base_url,
            api_provider=requested_provider,
        )
        gpt_time_ms = round((time.perf_counter() - generation_started) * 1000, 2)

        self._ensure_orcd_model_loaded()
        assert torch is not None
        assert self._orcd_model_bundle is not None

        agree_reason = self._build_generate_and_predict_reason(scores["agree_score"], True)
        disagree_reason = self._build_generate_and_predict_reason(scores["disagree_score"], True)

        content_ids = torch.tensor([self._tokenize_gnp(text)], device=self._device).long()
        pos_ids = torch.tensor([self._tokenize_gnp(agree_reason)], device=self._device).long()
        neg_ids = torch.tensor([self._tokenize_gnp(disagree_reason)], device=self._device).long()

        bundle = self._orcd_model_bundle
        inference_started = time.perf_counter()
        with torch.no_grad():
            content = bundle["bert"](content_ids)
            positive = bundle["bert2"](pos_ids)
            negative = bundle["bert3"](neg_ids)

            (pos_reason2text, pos_text2reason, positive,
             neg_reason2text, neg_text2reason, negative) = bundle["attention"](
                 content, positive, negative
             )

            _, r2t_aligned_agr, _ = bundle["r2t_usefulness"](content, pos_reason2text)
            _, t2r_aligned_agr, _ = bundle["t2r_usefulness"](content, pos_text2reason)
            _, r_aligned_agr, _ = bundle["reason_usefulness"](content, positive)
            _, r2t_aligned_dis, _ = bundle["r2t_usefulness"](content, neg_reason2text)
            _, t2r_aligned_dis, _ = bundle["t2r_usefulness"](content, neg_text2reason)
            _, r_aligned_dis, _ = bundle["reason_usefulness"](content, negative)

            final_feature = bundle["aggregator"](
                content,
                r2t_aligned_agr, t2r_aligned_agr, r_aligned_agr,
                r2t_aligned_dis, t2r_aligned_dis, r_aligned_dis,
            )
            logits = bundle["detection_module"](final_feature)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred].item())

        model_time_ms = round((time.perf_counter() - inference_started) * 1000, 2)

        return {
            "label": pred,
            "confidence": confidence,
            "model": model_key,
            "api_model_used": api_model,
            "raw_response": json.dumps({
                "original_score": scores["original_score"],
                "agree_score": scores["agree_score"],
                "disagree_score": scores["disagree_score"],
            }, ensure_ascii=False),
            "telemetry": {
                "gpt35_generation_ms": gpt_time_ms,
                "model_inference_ms": model_time_ms,
            },
        }
