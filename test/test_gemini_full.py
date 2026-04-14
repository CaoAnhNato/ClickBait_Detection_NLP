import os
import json
from pathlib import Path
from GUI.test.model_service import ClickbaitModelService

service = ClickbaitModelService(weights_root=Path("/mnt/c/Users/Admin/HUIT - Học Tập/Năm 3/Semester_2/Class/NLP/ORCD/GPT_3.5/weight"), preload_local_models=False)
result = service.predict(
    text="With Fine Print, the Rollout Dazzles This Time",
    model_key="orcd-gpt35",
    api_key="dummy",
    custom_api_model="gemini-3.1-flash-lite-preview",
    custom_api_provider="gemini",
    custom_api_base="https://api-v2.shopaikey.com/v1"
)

print(json.dumps(result.get("api_log", "NO_LOG"), indent=2))
