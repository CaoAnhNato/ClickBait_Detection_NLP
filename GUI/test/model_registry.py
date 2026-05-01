from pathlib import Path

MODEL_REGISTRY = {
    "bart-mnli": {
        "label": "Bart-mnli",
        "backend": "local",
        "weights_dir": "facebook_bart-large-mnli",
        "max_length": 128,
    },
    "roberta": {
        "label": "RoBerta",
        "backend": "local",
        "weights_dir": "FacebookAI_roberta-large",
        "max_length": 100,
    },
    "ggbert": {
        "label": "ggBert",
        "backend": "local",
        "weights_dir": "google-bert_bert-large-cased-whole-word-masking",
        "max_length": 100,
    },
    "sheepdog": {
        "label": "SheepDog",
        "backend": "sheepdog",
        "checkpoint_pattern": "clickbait_iter*.m",
        "max_length": 512,
    },
    "gpt4o-zero": {
        "label": "GPT4o - Zero",
        "backend": "gpt",
        "api_model": "gpt-4o",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0,
        "max_tokens": 150,
        "few_shot_k": 0,
    },
    "gpt4o-few": {
        "label": "GPT4o - Few",
        "backend": "gpt",
        "api_model": "gpt-4o",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0,
        "max_tokens": 150,
        "few_shot_k": 5,
    },
    "gemini-zero": {
        "label": "Gemini - Zero",
        "backend": "gpt",
        "api_model": "gemini-2.0-flash",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0,
        "max_tokens": 150,
        "few_shot_k": 0,
    },
    "qwen-zero": {
        "label": "Qwen - Zero",
        "backend": "gpt",
        "api_model": "qwen-plus",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0,
        "max_tokens": 150,
        "few_shot_k": 0,
    },
    "llama-zero": {
        "label": "Llama - Zero",
        "backend": "gpt",
        "api_model": "llama-3.1-70b-instruct",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0,
        "max_tokens": 150,
        "few_shot_k": 0,
    },
    "orcd-gpt35": {
        "label": "ORCD (GPT3.5)",
        "backend": "orcd",
        "api_model": "gpt-5.4-nano-2026-03-17",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0.3,
        "temperature_score": 0.0,
        "temperature_reasoning": 0.3,
        "max_tokens_score": 16,
        "max_tokens_reasoning": 150,
        "max_retries_score": 1,
        "max_retries_reasoning": 3,
        "timeout_score_s": 3.0,
        "timeout_reasoning_s": 20.0,
        "max_agree_iterations": 10,
        "max_disagree_iterations": 8,
        "disagree_analysis_every_n": 3,
        "enable_payload_preview": False,
    },
    "orcd-gpt4o": {
        "label": "ORCD (GPT4o)",
        "backend": "orcd",
        "api_model": "gpt-4o-2024-11-20",
        "base_url": "https://api-v2.shopaikey.com/v1",
        "temperature": 0.3,
        "temperature_score": 0.0,
        "temperature_reasoning": 0.3,
        "max_tokens_score": 16,
        "max_tokens_reasoning": 150,
        "max_retries_score": 1,
        "max_retries_reasoning": 2,
        "timeout_score_s": 3.0,
        "timeout_reasoning_s": 20.0,
        "max_agree_iterations": 8,
        "max_disagree_iterations": 6,
        "disagree_analysis_every_n": 3,
        "enable_payload_preview": False,
    },
    "generate-and-predict": {
        "label": "Generate & Predict (GPT3.5 + ModelBART)",
        "backend": "generate_and_predict",
        "api_model": "gpt-3.5-turbo-1106",
        "base_url": "https://api-v2.shopaikey.com/v1",
    },
}


def get_weights_root() -> Path:
    start_dir = Path(__file__).resolve().parent

    # Auto-discover workspace root by looking for Bert_Fami/weights upward.
    for candidate in [start_dir, *start_dir.parents]:
        weights_dir = candidate / "Bert_Fami" / "weights"
        if weights_dir.exists():
            return weights_dir

    # Fallback to expected layout: <workspace>/GUI/test/model_registry.py
    if len(start_dir.parents) > 1:
        return start_dir.parents[1] / "Bert_Fami" / "weights"
    return start_dir / "Bert_Fami" / "weights"
