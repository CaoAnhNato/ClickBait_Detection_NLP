import time
import os
import re
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# =================================================================================
# QUICK CONFIGURATION (Thiết lập sẵn tại đây)
# =================================================================================
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v" # Điền API Key của bạn vào đây
BASE_URL = "https://api-v2.shopaikey.com/v1"
MODEL_NAME = "gpt-3.5-turbo-1106"

# =================================================================================
# GLOBAL CONFIG
# =================================================================================
CONFIG = {
    'base_url': BASE_URL,
    'api_key': API_KEY,
    'model': MODEL_NAME,
    'max_retries': 3,
    'temp_score': 0.1,
    'temp_reasoning': 0.3,
    'top_p_score': 1.0,
    'top_p_reasoning': 1.0,
    'max_tokens_score': 100,
    'max_tokens_reasoning': 150
}

# =================================================================================
# PROMPTS
# =================================================================================
prompt1 = "Goal: As a news expert, score the title's content to determine its accuracy and completeness, and assess people's agreement with the title.\nRequirement 1: The title content is {}.\nRequirement 2: The score range is from 0 to 100, where 0 means complete disagreement, 50 means difficult to judge, and 100 means complete agreement. The score should be humanized and not restricted to multiples of 5 and strictly a single integer number.\nRequirement 3: Output format[int].\n"
prompt_reassign = "Goal: Re-assess the agreement level based on the title's content.Requirement 1: The content of the title is {}.\nRequirement 2: Consider the previous agreement score for the title, which was {}.\nRequirement 3: The new score should fall within the range of {} to {}.\nRequirement 4: The score should be between 0 and 100 and not restricted to multiples of 5 and strictly a single integer number.\nRequirement 5: The output format is [int]."
prompt2 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people believe the content in the title.\nRequirement 1: The title content is {}.\nRequirement 2: Please agree with the title content in combination with the following four aspects:\n1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n2. Logic: Are there any leaps in reasoning or inconsistencies?\n3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\nRequirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\nRequirement 4: The output format is [reasoning content].\n"
prompt3 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people disbelieve the content in the title.\nRequirement 1: The title content is {}.\nRequirement 2: Please disagree with the title content in combination with the following four aspects:\n1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n2. Logic: Are there any leaps in reasoning or inconsistencies?\n3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\nRequirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\nRequirement 4: The output format is [reasoning content].\n"
prompt4 = 'Goal:  Re-score based on the title content, initial score, and {}reasoning.Requirement 1: The title is {}.\nRequirement 2: The initial score is {}.\nRequirement 3: The {} reasoning content is {}.\nRequirement 4: The score should be between 0 and 100 and not restricted to multiples of 5 and strictly a single integer number.\nRequirement 5: The output format is [int].\n'
prompt5 = 'Goal: Analyze the {} reasoning content from the perspectives of rationality and logic.\nRequirement 1: Consider the previous {} reasoning content: {}\nRequirement 2: Consider the previous score based on the {} reasoning: {}.\nRequirement 3: The analysis should be limited to 50-70 words.\nRequirement 4: Output format [reasoning content].\n'
prompt6 = "Goal: Regenerate {} reasoning content, because the previous reasoning did not effectively {} the title's recognition score\nRequirement 1: The title is {}.\nRequirement 2: The initial score is {}.\nRequirement 3: Consider the previous {} reasoning content: {}.\nRequirement 4: Consider the title score based on the previous {} reasoning: {}.\nRequirement 5: Consider the evaluation of the reasoning for {}: {}. \nRequirement 6: Analyze the logical inconsistencies in the previous reasoning and explain why the new reasoning is more suitable for the title content.\nRequirement 7: New inference generation should combine the following four aspects and adapt to the content of the title to {} people's identification with the content of the title and make people {} in the content of the title.1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n2. Logic: Are there any leaps in reasoning or inconsistencies?\n3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\nRequirement 8: The limit for inference is 40-60 words, and the limit for explanation is 20-40 words. The inference content is placed in [] and the explanation content is placed in ().\nRequirement 9: Output format is [Reasoning Content] (Explanatory Content).\nRequirement 10: The score should still range from 0 to 100, and it should be more humanized, not restricted to multiples of 5 and strictly a single integer number.\nRequirement 11: Output format for the score is [int].\n"

# =================================================================================
# UTILS
# =================================================================================
def create_openai_client(base_url=None, api_key=None):
    return OpenAI(
        base_url=base_url or CONFIG['base_url'],
        api_key=api_key or CONFIG['api_key']
    )

def generate_score(prompt, client=None):
    if client is None: client = create_openai_client()
    max_retries = CONFIG['max_retries']
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=CONFIG['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful and expert news analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=CONFIG['max_tokens_score'],
                temperature=CONFIG['temp_score'],
                top_p=CONFIG['top_p_score']
            )
            content = response.choices[0].message.content
            if content is None: raise ValueError("None content")
            match = re.search(r'\d+', content.strip())
            if match: return max(0, min(100, int(match.group())))
        except Exception as e:
            time.sleep(1)
        attempt += 1
    return 50

def generate_res(prompt, client=None):
    if client is None: client = create_openai_client()
    max_retries = CONFIG['max_retries']
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=CONFIG['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful and expert news analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=CONFIG['max_tokens_reasoning'],
                temperature=CONFIG['temp_reasoning'],
                top_p=CONFIG['top_p_reasoning']
            )
            content = response.choices[0].message.content
            if content and content.strip(): return content.strip()
        except Exception as e:
            time.sleep(1)
        attempt += 1
    return "Error generating response"

def extract_quoted_text(text, input_text):
    old_text = text
    sentences_to_remove = ["agree reasoning", "disagree reasoning", "Clickbait", "Non-clickbait", "increase", "lower"]
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, text)
    if matches: text = ''.join(matches)
    text = re.sub(r'[\'"]', '', text)
    input_text_clean = re.sub(r'[\'"]', '', input_text)
    if text.split()[:10] == input_text_clean.split()[:10]:
        text = old_text
    for sentence in sentences_to_remove:
        text = re.sub(re.escape(sentence) + r'\s*', '', text)
    return text

def process_single_row(title_text, client):
    input_text = str(title_text)
    str3, str4, str5, str6, str7, str8 = 'agree', 'disagree', 'increase', 'lower', 'believe', 'disbelieve'
    
    # --- Định nghĩa các Task chạy song song ---
    def get_original_score():
        score = generate_score(prompt1.format(input_text), client)
        while score < 30 or score > 70:
            score = generate_score(prompt_reassign.format(input_text, score, 30, 70), client)
        return score
        
    def get_agree_reason():
        return extract_quoted_text(generate_res(prompt2.format(input_text), client), input_text)
        
    def get_disagree_reason():
        return extract_quoted_text(generate_res(prompt3.format(input_text), client), input_text)

    # 1. Gửi ĐỒNG THỜI 3 requests khởi tạo (Original Score, Agree Reason, Disagree Reason)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_score = executor.submit(get_original_score)
        future_agree = executor.submit(get_agree_reason)
        future_disagree = executor.submit(get_disagree_reason)
        
        # Đợi và lấy kết quả
        original_score = future_score.result()
        agree_reason = future_agree.result()
        disagree_reason = future_disagree.result()

    # --- Định nghĩa flow cho Agree và Disagree để chạy song song chặng cuối ---
    def process_agree_flow(reason, ref_score):
        agr_scr = generate_score(prompt4.format(str3, input_text, ref_score, str3, reason), client)
        count1 = 0
        while (agr_scr - ref_score < 10 or agr_scr <= 55) and count1 < 2:
            ret_agree_reason = generate_res(prompt5.format(str3, str3, reason, str3, agr_scr), client)
            reason = extract_quoted_text(generate_res(prompt6.format(str3, str5, input_text, ref_score, str3, reason, str3, agr_scr, str3, ret_agree_reason, str5, str7), client), input_text)
            agr_scr = generate_score(prompt6.format(str3, str5, input_text, ref_score, str3, reason, str3, agr_scr, str3, ret_agree_reason, str5, str7), client)
            count1 += 1
        return agr_scr

    def process_disagree_flow(reason, ref_score):
        dis_scr = generate_score(prompt4.format(str4, input_text, ref_score, str4, reason), client)
        count2 = 0
        while (ref_score - dis_scr < 10 or dis_scr >= 45) and count2 < 2:
            ret_disagree_reason = generate_res(prompt5.format(str4, str4, reason, str4, dis_scr), client)
            reason = extract_quoted_text(generate_res(prompt6.format(str4, str6, input_text, ref_score, str4, reason, str4, dis_scr, str4, ret_disagree_reason, str6, str8), client), input_text)
            dis_scr = generate_score(prompt6.format(str4, str6, input_text, ref_score, str4, reason, str4, dis_scr, str4, ret_disagree_reason, str6, str8), client)
            count2 += 1
        return dis_scr

    # 2. Xử lý song song luồng tính Agree Code và Disagree Score (vì 2 luồng này hiện chỉ cần dựa vào Original Score)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_agr_score = executor.submit(process_agree_flow, agree_reason, original_score)
        future_dis_score = executor.submit(process_disagree_flow, disagree_reason, original_score)
        
        agr_score = future_agr_score.result()
        dis_score = future_dis_score.result()

    return {
        'original_score': original_score,
        'agree_score': agr_score,
        'disagree_score': dis_score,
    }

def main():
    print("=== SORG DATA GENERATION SPEED TEST ===")
    
    # Yêu cầu người dùng điền key nếu chưa thiết lập tại block "QUICK CONFIGURATION"
    if not CONFIG['api_key']:
        config_key = input("Enter API Key (Bị trống do chưa thiết lập API_KEY ở dòng 10): ").strip()
        if config_key:
            CONFIG['api_key'] = config_key
        else:
            print("API Key is required! Vui lòng set biến `API_KEY` ở đầu file hoặc nhập trực tiếp khi dùng.")
            return

    # Base url và model luôn được set do đã lấy từ biến global trước đó
    print(f"[*] Đang sử dụng API: {CONFIG['base_url']}")
    print(f"[*] Đang sử dụng Model: {CONFIG['model']}")

    client = create_openai_client()
    
    while True:
        title = input("\nEnter Title to test (hoặc nhập 'q', 'quit' để thoát): ").strip()
        if title.lower() in ['q', 'quit']:
            print("Thoát chương trình!")
            break
        elif not title:
            print("Title không hợp lệ!")
            continue
        
        print(f"\n[+] Bắt đầu sinh dữ liệu cho title: '{title}' bằng model {CONFIG['model']} ...")
        start_time = time.time()
        
        try:
            result = process_single_row(title, client)
            elapsed_time = time.time() - start_time
            print(f"\n[+] Kết quả:")
            print(f"    - Original Score: {result['original_score']}")
            print(f"    - Agree Score: {result['agree_score']}")
            print(f"    - Disagree Score: {result['disagree_score']}")
            print(f"\n=> Tổng thời gian sinh data cho '{title}': {elapsed_time:.2f} giây")
            print("-" * 50)
        except Exception as e:
            print(f"\n[-] Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
