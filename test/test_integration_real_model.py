import os
from pathlib import Path

import pytest

from GUI.test.model_service import ClickbaitModelService


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to run real model integration test",
)


def test_real_model_bart_inference_runs_end_to_end():
    root = Path(__file__).resolve().parents[1]
    service = ClickbaitModelService(weights_root=root / "Bert_Fami" / "weights")

    result = service.predict(
        text="You won't believe what happened next in this story",
        model_key="bart-mnli",
    )

    assert result["label"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model"] == "bart-mnli"
