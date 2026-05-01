"""
Interactive Clickbait Predictor: generates scores via GPT-3.5 and predicts with ModelBART.

Usage:
  - Enter a news headline when prompted.
  - The script generates agree/disagree scores via GPT-3.5, then predicts
    Clickbait (1) or Non-Clickbait (0) using ModelBART.
  - Type 'q' and press Enter to quit.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openai import AsyncOpenAI
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ORCD_GPT35 = ROOT / "ORCD" / "GPT_3.5"
sys.path.insert(0, str(ORCD_GPT35))
sys.path.insert(0, str(ORCD_GPT35 / "train"))

# ---------------------------------------------------------------------------
# GPT-3.5 Configuration
# ---------------------------------------------------------------------------
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BASE_URL = "https://api-v2.shopaikey.com/v1"
MODEL_NAME = "gpt-3.5-turbo-1106"

# ---------------------------------------------------------------------------
# ModelBART Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
MAX_LEN = 100
WEIGHT_PATH = str(ORCD_GPT35 / "weight" / "best_teachermodel.pth")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------------------------------------------------------------
# Step 1 – GPT-3.5 score generation (from test_sorg1_fast_pipeline.py)
# ---------------------------------------------------------------------------
async def generate_scores_and_reasons(client: AsyncOpenAI, title: str) -> dict:
    """Call GPT-3.5 to generate original/agree/disagree scores."""
    prompt = f"""
Goal: As a news expert, evaluate the title's content and score it according to the criteria below.
Requirement 1: The title is '{title}'.
Requirement 2: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity.
Requirement 3: First, assign an "original_score" representing the general public's agreement/belief level with the title (30 to 70).
Requirement 4: Then, formulate an "agree_reason" (40-60 words) that advocates for the title being completely truthful. Based on this, assign an "agree_score" that must be at least 15 points higher than original_score (up to 100).
Requirement 5: Finally, formulate a "disagree_reason" (40-60 words) that highlights any illogical leaps or vague language. Based on this, assign a "disagree_score" that must be at least 15 points lower than original_score (down to 0).
Requirement 6: All scores should be strictly single integers.
Requirement 7: The output MUST be a valid JSON object with EXACTLY three numeric fields: "original_score", "agree_score", "disagree_score". Do not output the reason text, just the final scores.
"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        return {
            "original_score": int(data.get("original_score", 50)),
            "agree_score": int(data.get("agree_score", 80)),
            "disagree_score": int(data.get("disagree_score", 20)),
        }
    except Exception:
        return {"original_score": 50, "agree_score": 85, "disagree_score": 35}


def build_reason_text(score: int, is_clickbait: bool) -> str:
    """Build a synthetic reason string matching the CSV column format."""
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


# ---------------------------------------------------------------------------
# Step 2 – ModelBART components (from inference_test.py)
# ---------------------------------------------------------------------------
from train.modelbart import (
    Similarity,
    DetectionModule,
    Attention_Encoder,
    Reason_Similarity,
    Aggregator,
    BertEncoder,
)


def tokenize_and_numericalize_data(text, tokenizer):
    tokenized = tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_LEN
    )
    return tokenized["input_ids"]


class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.csv_data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        try:
            text = str(self.csv_data.loc[idx, "title"])
            pos = str(self.csv_data.loc[idx, "agree_reason"])
            neg = str(self.csv_data.loc[idx, "disagree_reason"])

            content_input_id = tokenize_and_numericalize_data(text, self.tokenizer)
            pos_input_id = tokenize_and_numericalize_data(pos, self.tokenizer)
            neg_input_id = tokenize_and_numericalize_data(neg, self.tokenizer)

            agree_score = float(self.csv_data.loc[idx, "agree_score"])
            disagree_score = float(self.csv_data.loc[idx, "disagree_score"])
            label = int(self.csv_data.loc[idx, "label"])

            return {
                "content": torch.tensor(content_input_id),
                "pos_reason": torch.tensor(pos_input_id),
                "neg_reason": torch.tensor(neg_input_id),
                "label": torch.tensor(label),
                "agree_soft_label": torch.tensor(agree_score / 100, dtype=torch.float32),
                "disagree_soft_label": torch.tensor(
                    disagree_score / 100, dtype=torch.float32
                ),
            }
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def load_model(weight_path):
    bert = BertEncoder(256, False).to(device)
    bert2 = BertEncoder(256, False).to(device)
    bert3 = BertEncoder(256, False).to(device)
    attention = Attention_Encoder().to(device)
    R2T_usefulness = Similarity().to(device)
    T2R_usefulness = Similarity().to(device)
    Reason_usefulness = Reason_Similarity().to(device)
    aggregator = Aggregator().to(device)
    detection_module = DetectionModule().to(device)

    if os.path.exists(weight_path):
        print(f"  Loading weights from: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        if isinstance(checkpoint, dict) and "bert" in checkpoint:
            bert.load_state_dict(checkpoint["bert"])
            bert2.load_state_dict(checkpoint["bert2"])
            bert3.load_state_dict(checkpoint["bert3"])
            attention.load_state_dict(checkpoint["attention"])
            R2T_usefulness.load_state_dict(checkpoint["R2T_usefulness"])
            T2R_usefulness.load_state_dict(checkpoint["T2R_usefulness"])
            Reason_usefulness.load_state_dict(checkpoint["Reason_usefulness"])
            aggregator.load_state_dict(checkpoint["aggregator"])
            detection_module.load_state_dict(checkpoint["detection_module"])
            print("  Weights loaded (separate module state_dicts).")
        else:
            print("  Warning: unknown checkpoint format, using random weights.")
    else:
        print(f"  Weight file not found: {weight_path}  |  using random weights.")

    for m in [bert, bert2, bert3, attention,
              R2T_usefulness, T2R_usefulness, Reason_usefulness,
              aggregator, detection_module]:
        m.eval()

    return (bert, bert2, bert3, attention,
            R2T_usefulness, T2R_usefulness, Reason_usefulness,
            aggregator, detection_module)


def predict_one(models, dataloader):
    """Return (prediction, label) for the single item in dataloader."""
    (bert, bert2, bert3, attention,
     R2T_usefulness, T2R_usefulness, Reason_usefulness,
     aggregator, detection_module) = models

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                return None, None

            news_content = batch["content"].to(device).long()
            pos = batch["pos_reason"].to(device).long()
            neg = batch["neg_reason"].to(device).long()
            label = batch["label"].to(device)

            content = bert(news_content)
            positive = bert2(pos)
            negative = bert3(neg)

            (pos_reason2text, pos_text2reason, positive,
             neg_reason2text, neg_text2reason, negative) = attention(
                 content, positive, negative
             )

            _, R2T_aligned_agr, _ = R2T_usefulness(content, pos_reason2text)
            _, T2R_aligned_agr, _ = T2R_usefulness(content, pos_text2reason)
            _, R_aligned_agr, _ = Reason_usefulness(content, positive)
            _, R2T_aligned_dis, _ = R2T_usefulness(content, neg_reason2text)
            _, T2R_aligned_dis, _ = T2R_usefulness(content, neg_text2reason)
            _, R_aligned_dis, _ = Reason_usefulness(content, negative)

            final_feature = aggregator(
                content, R2T_aligned_agr, T2R_aligned_agr, R_aligned_agr,
                R2T_aligned_dis, T2R_aligned_dis, R_aligned_dis
            )
            pre_detection = detection_module(final_feature)

            pred = pre_detection.argmax(1).item()
            return pred, label.item()


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Clickbait Predictor  |  GPT-3.5 + ModelBART")
    print("=" * 60)
    print("  Enter a news headline to classify.")
    print("  Type 'q' and press Enter to quit.\n")

    # Load model ONCE at startup
    print("[*] Loading ModelBART ...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    models = load_model(WEIGHT_PATH)
    print("[*] Model ready.\n")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    while True:
        try:
            title = input("Headline >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if title.lower() == "q":
            print("Goodbye!")
            break

        if not title:
            print("  (empty input, please enter a headline)\n")
            continue

        # Step A: Generate scores via GPT-3.5
        print(f"  Processing: \"{title[:70]}{'...' if len(title) > 70 else ''}\"")
        t0 = time.perf_counter()
        scores = asyncio.run(generate_scores_and_reasons(client, title))
        elapsed = time.perf_counter() - t0
        print(f"  GPT-3.5 scores  ({elapsed:.2f}s): "
              f"original={scores['original_score']}, "
              f"agree={scores['agree_score']}, "
              f"disagree={scores['disagree_score']}")

        # Step B: Build single-row DataFrame
        # label=1 is a placeholder so FakeNewsDataset doesn't crash;
        # the model prediction is what we actually care about.
        row = {
            "title": title,
            "label": 1,
            "agree_reason": build_reason_text(scores["agree_score"], True),
            "disagree_reason": build_reason_text(scores["disagree_score"], True),
            "agree_score": scores["agree_score"],
            "disagree_score": scores["disagree_score"],
            "original_score": scores["original_score"],
            "agree_reason_all": "[]",
            "disagree_reason_all": "[]",
            "agree_score_all": scores["agree_score"],
            "disagree_score_all": scores["disagree_score"],
        }
        df = pd.DataFrame([row])

        # Step C: Predict
        dataset = FakeNewsDataset(df, tokenizer, MAX_LEN)
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
        )
        pred, _ = predict_one(models, dataloader)

        # Step D: Show result
        label_map = {0: "Non-Clickbait", 1: "Clickbait"}
        result = label_map.get(pred, "Unknown")
        print(f"  --> Prediction: [{result}]  ({result})\n")


if __name__ == "__main__":
    main()
