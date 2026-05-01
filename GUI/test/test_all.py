import sys
import os
import time
import asyncio
import json
import torch
import numpy as np

# Append ORCD/GPT_3.5 to sys.path so we can import train.modelbart
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ORCD/GPT_3.5')))

from transformers import BertTokenizer
from train.modelbart import Similarity, DetectionModule, Attention_Encoder, Reason_Similarity, Aggregator, BertEncoder
from openai import AsyncOpenAI

# =================================================================================
# QUICK CONFIGURATION
# =================================================================================
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BASE_URL = "https://api-v2.shopaikey.com/v1"
MODEL_NAME = "gpt-3.5-turbo-1106"
WEIGHT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ORCD/GPT_3.5/weight/best_teachermodel.pth'))
MAX_LEN = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_async_openai_client():
    return AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

async def process_single_row(title: str, client: AsyncOpenAI) -> dict:
    prompt = f"""
Goal: As a news expert, evaluate the title's content and score it according to the criteria below.
Requirement 1: The title is '{title}'.
Requirement 2: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity.
Requirement 3: First, assign an "original_score" representing the general public's agreement/belief level with the title (30 to 70).
Requirement 4: Then, formulate an "agree_reason" (40-60 words) that advocates for the title being completely truthful. Based on this, assign an "agree_score" that must be at least 15 points higher than original_score (up to 100).
Requirement 5: Finally, formulate a "disagree_reason" (40-60 words) that highlights any illogical leaps or vague language. Based on this, assign a "disagree_score" that must be at least 15 points lower than original_score (down to 0).
Requirement 6: All scores should be formatted as strictly single integers.
Requirement 7: The output MUST be a valid JSON object with EXACTLY five fields: "original_score", "agree_score", "disagree_score", "agree_reason", "disagree_reason".
"""
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        response_format={ "type": "json_object" },
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=250,
    )
    res_content = response.choices[0].message.content
    try:
        data = json.loads(res_content)
        return {
            'original_score': int(data.get('original_score', 50)),
            'agree_score': int(data.get('agree_score', 80)),
            'disagree_score': int(data.get('disagree_score', 20)),
            'agree_reason': str(data.get('agree_reason', '')),
            'disagree_reason': str(data.get('disagree_reason', ''))
        }
    except Exception as e:
        return {'original_score': 50, 'agree_score': 85, 'disagree_score': 35, 'agree_reason': '', 'disagree_reason': ''}


def load_model_weights(weight_path):
    bert = BertEncoder(256, False).to(device)
    bert2 = BertEncoder(256, False).to(device)
    bert3 = BertEncoder(256, False).to(device)
    attention = Attention_Encoder().to(device)
    R2T_usefulness = Similarity().to(device)
    T2R_usefulness = Similarity().to(device)
    Reason_usefulness = Reason_Similarity().to(device)
    aggregator = Aggregator().to(device)
    detection_module = DetectionModule().to(device)

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'bert' in checkpoint:
            bert.load_state_dict(checkpoint['bert'])
            bert2.load_state_dict(checkpoint['bert2'])
            bert3.load_state_dict(checkpoint['bert3'])
            attention.load_state_dict(checkpoint['attention'])
            R2T_usefulness.load_state_dict(checkpoint['R2T_usefulness'])
            T2R_usefulness.load_state_dict(checkpoint['T2R_usefulness'])
            Reason_usefulness.load_state_dict(checkpoint['Reason_usefulness'])
            aggregator.load_state_dict(checkpoint['aggregator'])
            detection_module.load_state_dict(checkpoint['detection_module'])
        elif 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
            pass # Simplified for test
        else:
            try:
                bert_dict = {k.replace('bert.', ''): v for k, v in checkpoint.items() if k.startswith('bert.') and not k.startswith('bert2.') and not k.startswith('bert3.')}
                bert2_dict = {k.replace('bert2.', ''): v for k, v in checkpoint.items() if k.startswith('bert2.')}
                bert3_dict = {k.replace('bert3.', ''): v for k, v in checkpoint.items() if k.startswith('bert3.')}
                attention_dict = {k.replace('attention.', ''): v for k, v in checkpoint.items() if k.startswith('attention.')}
                r2t_dict = {k.replace('R2T_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('R2T_usefulness.')}
                t2r_dict = {k.replace('T2R_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('T2R_usefulness.')}
                reason_dict = {k.replace('Reason_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('Reason_usefulness.')}
                agg_dict = {k.replace('aggregator.', ''): v for k, v in checkpoint.items() if k.startswith('aggregator.')}
                det_dict = {k.replace('detection_module.', ''): v for k, v in checkpoint.items() if k.startswith('detection_module.')}
                if bert_dict: bert.load_state_dict(bert_dict, strict=False)
                if bert2_dict: bert2.load_state_dict(bert2_dict, strict=False)
                if bert3_dict: bert3.load_state_dict(bert3_dict, strict=False)
                if attention_dict: attention.load_state_dict(attention_dict, strict=False)
                if r2t_dict: R2T_usefulness.load_state_dict(r2t_dict, strict=False)
                if t2r_dict: T2R_usefulness.load_state_dict(t2r_dict, strict=False)
                if reason_dict: Reason_usefulness.load_state_dict(reason_dict, strict=False)
                if agg_dict: aggregator.load_state_dict(agg_dict, strict=False)
                if det_dict: detection_module.load_state_dict(det_dict, strict=False)
            except Exception:
                pass

    bert.eval()
    bert2.eval()
    bert3.eval()
    attention.eval()
    R2T_usefulness.eval()
    T2R_usefulness.eval()
    Reason_usefulness.eval()
    aggregator.eval()
    detection_module.eval()

    return bert, bert2, bert3, attention, R2T_usefulness, T2R_usefulness, Reason_usefulness, aggregator, detection_module

def tokenize_and_numericalize_data(text, tokenizer):
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN)
    return tokenized['input_ids']

def run_model_inference(models, tokenizer, title, result):
    bert, bert2, bert3, attention, R2T_usefulness, T2R_usefulness, Reason_usefulness, aggregator, detection_module = models

    content_input_id = tokenize_and_numericalize_data(title, tokenizer)
    pos_input_id = tokenize_and_numericalize_data(result['agree_reason'], tokenizer)
    neg_input_id = tokenize_and_numericalize_data(result['disagree_reason'], tokenizer)

    content = torch.tensor([content_input_id]).to(device).long()
    pos = torch.tensor([pos_input_id]).to(device).long()
    neg = torch.tensor([neg_input_id]).to(device).long()

    with torch.no_grad():
        c = bert(content)
        p = bert2(pos)
        n = bert3(neg)

        pos_reason2text, pos_text2reason, p, neg_reason2text, neg_text2reason, n = attention(c, p, n)

        text_R2T_aligned_agr, R2T_aligned_agr, _ = R2T_usefulness(c, pos_reason2text)
        text_T2R_aligned_agr, T2R_aligned_agr, _ = T2R_usefulness(c, pos_text2reason)
        text_R_aligned_agr, R_aligned_agr, _ = Reason_usefulness(c, p)

        text_R2T_aligned_dis, R2T_aligned_dis, _ = R2T_usefulness(c, neg_reason2text)
        text_T2R_aligned_dis, T2R_aligned_dis, _ = T2R_usefulness(c, neg_text2reason)
        text_R_aligned_dis, R_aligned_dis, _ = Reason_usefulness(c, n)

        final_feature = aggregator(
            c, R2T_aligned_agr, T2R_aligned_agr, R_aligned_agr,
            R2T_aligned_dis, T2R_aligned_dis, R_aligned_dis
        )
        pre_detection = detection_module(final_feature)
        prediction = pre_detection.argmax(1).cpu().item()
        
    return prediction

async def main():
    print("=== FULL PIPELINE (GENERATION + INFERENCE) ===")
    
    print("Loading models and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    models = load_model_weights(WEIGHT_PATH)
    client = create_async_openai_client()
    print("Models loaded successfully.")
    
    while True:
        title = input("\nEnter Title to test (hoặc nhập 'q', 'quit' để thoát): ").strip()
        if title.lower() in ['q', 'quit']:
            print("Thoát chương trình!")
            break
        elif not title:
            continue
            
        print(f"\n[+] Processing title: '{title}'")
        iter_start_time = time.perf_counter()
        
        # Generation step
        gen_start_time = time.perf_counter()
        result = await process_single_row(title, client)
        gen_time = time.perf_counter() - gen_start_time
        
        # Inference step
        prediction = run_model_inference(models, tokenizer, title, result)
        
        elapsed_time = time.perf_counter() - iter_start_time
        
        print(f"    => Generation Data: Orig: {result['original_score']} | Agree: {result['agree_score']} | Disagree: {result['disagree_score']}")
        print(f"    => Predicted Label: {prediction} ({'Clickbait' if prediction == 1 else 'Non-Clickbait'})")
        print(f"    ⏱ Generation Time: {gen_time:.2f} s | Total Step Time: {elapsed_time:.2f} s")

if __name__ == "__main__":
    asyncio.run(main())
