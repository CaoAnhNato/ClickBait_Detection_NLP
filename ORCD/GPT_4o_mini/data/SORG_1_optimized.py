"""
=================================================================================
SCRIPT OPTIMIZED: Kết hợp Multiprocessing + Threading cho hệ thống phân loại Clickbait
=================================================================================
Kiến trúc:
    - MULTIPROCESSING: Chia dataset thành N phần, mỗi phần xử lý bởi 1 process độc lập
    - THREADING: Trong mỗi process, sử dụng ThreadPoolExecutor để xử lý đồng thời MAX_CONCURRENT=5 API calls
    - CHUNK PROCESSING: Xử lý theo batch CHUNK_SIZE=100 để tối ưu bộ nhớ
    - FILE MANAGEMENT: Không ghi file trong quá trình xử lý, chỉ ghi 1 lần cuối cùng
    - CLI ARGUMENTS: Tất cả tham số có thể điều chỉnh qua command line với tên viết tắt
=================================================================================
"""

import re
import time
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import argparse

# =================================================================================
# PHẦN 1: GLOBAL CONFIG - Sẽ được khởi tạo từ CLI arguments
# =================================================================================

# Global config dictionary sẽ được cập nhật từ argparse
CONFIG = {
    'base_url': "https://api-v2.shopaikey.com/v1",
    'api_key': "sk-HlkHAJijdlayeUE5y2pmyyIJgjZEK5Tzutrb8mvz29BWhDiE",
    'model': "gpt-4o-mini-2024-07-18",
    'max_retries': 5,
    'temp_score': 0.1,
    'temp_reasoning': 0.3,
    'top_p_score': 1.0,
    'top_p_reasoning': 1.0,
    'max_tokens_score': 100,
    'max_tokens_reasoning': 150
}

# =================================================================================
# PHẦN 2: TÍCH HỢP UTILS.PY - CÁC HÀM GỌI API
# =================================================================================

def create_openai_client(base_url=None, api_key=None):
    """
    Tạo OpenAI client mới cho mỗi process/thread để đảm bảo thread-safety

    Args:
        base_url: Base URL của API (lấy từ CONFIG nếu None)
        api_key: API key (lấy từ CONFIG nếu None)
    """
    return OpenAI(
        base_url=base_url or CONFIG['base_url'],
        api_key=api_key or CONFIG['api_key']
    )

def generate_score(prompt, client=None):
    """
    Hàm sinh ra điểm số, ép kiểu về Int, xử lý lỗi None và giới hạn trong khoảng 0-100.
    Thread-safe: Nhận client làm tham số hoặc tạo mới nếu không có.
    Các tham số (temperature, top_p, max_tokens, max_retries) được lấy từ CONFIG.
    """
    if client is None:
        client = create_openai_client()

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
            if content is None:
                raise ValueError("API trả về Content là None")

            text_response = content.strip()

            match = re.search(r'\d+', text_response)
            if match:
                score = int(match.group())
                return max(0, min(100, score))
            else:
                print(f"[Retry {attempt+1}/{max_retries}] Không tìm thấy số. Chuỗi trả về: '{text_response}'")
                finish_reason = response.choices[0].finish_reason
                print(f"Lý do API dừng: {finish_reason}")

        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Lỗi gọi API generate_score: {e}")
            time.sleep(1)  # Đợi 1 giây trước khi retry

        attempt += 1

    print(f"[CẢNH BÁO] generate_score đã thất bại sau {max_retries} lần thử. Trả về giá trị mặc định 50.")
    return 50

def generate_res(prompt, client=None):
    """
    Đảm bảo trả về một chuỗi văn bản, thử lại nếu là None.
    Thread-safe: Nhận client làm tham số hoặc tạo mới nếu không có.
    Các tham số (temperature, top_p, max_tokens, max_retries) được lấy từ CONFIG.
    """
    if client is None:
        client = create_openai_client()

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
            if content is not None and content.strip() != "":
                return content.strip()

            print(f"[Retry {attempt+1}/{max_retries}] Nội dung trống, đang thử lại...")

        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Lỗi API generate_res: {e}")
            time.sleep(1)  # Đợi 1 giây trước khi retry

        attempt += 1

    print(f"[CẢNH BÁO] generate_res đã thất bại sau {max_retries} lần thử. Trả về thông báo lỗi.")
    return "Error: Could not generate response"

# =================================================================================
# PHẦN 2: BẢO TOÀN LOGIC NGHIỆP VỤ - EXTRACT QUOTED TEXT
# =================================================================================

def extract_quoted_text(text, input_text):
    """Giữ nguyên 100% logic từ SORG_1.py"""
    old_text = text
    sentences_to_remove = [
        "agree reasoning",
        "disagree reasoning",
        "Clickbait",
        "Non-clickbait",
        "increase",
        "lower"
    ]
    sentences_to_remove2 = [
        "Here's another that can the of the :",
        "Here's a that can the of the :",
        "Here is a that can the of the :",
        "Here's another attempt at generating a that can the of the :",
        "Here's a new that can the of the :",
        "Here's a revised that can the of the :",
        "Here's a rewritten that can the of the :",
        "Here's a possible that can the of the :"
    ]

    pattern = r'<(.*?)>'
    matches = re.findall(pattern, text)

    if matches:
        text = ''.join(matches)

    cleaned_text = re.sub(r'[\'"]', '', text)
    wordsInText = re.split(r'\s+', cleaned_text.strip())
    words_in_text = wordsInText[:10]

    cleaned_text = re.sub(r'[\'"]', '', input_text)
    wordsInText = re.split(r'\s+', cleaned_text.strip())
    words_in_input_text = wordsInText[:10]

    if words_in_text == words_in_input_text:
        text = old_text

    for sentence in sentences_to_remove:
        pattern = re.escape(sentence) + r'\s*'
        text = re.sub(pattern, '', text)

    for sentence2 in sentences_to_remove2:
        pattern = re.escape(sentence2) + r'\s*'
        text = re.sub(pattern, '', text)

    print('处理后：', text)
    return text

# =================================================================================
# PHẦN 3: BẢO TOÀN LOGIC NGHIỆP VỤ - TOÀN BỘ PROMPTS
# =================================================================================

# Giữ nguyên 100% các prompts từ dòng 79-146 SORG_1.py
prompt1 = "Goal: As a news expert, score the title's content to determine its accuracy and completeness, and assess people's agreement with the title.\n"
prompt1 += 'Requirement 1: The title content is {}.\n'
prompt1 += 'Requirement 2: The score range is from 0 to 100, where 0 means complete disagreement, 50 means difficult to judge, and 100 means complete agreement. The score should be humanized and not restricted to multiples of 5 and strictly a single integer number.\n'
prompt1 += 'Requirement 3: Output format[int].\n'

prompt_reassign = "Goal: Re-assess the agreement level based on the title's content."
prompt_reassign += 'Requirement 1: The content of the title is {}.\n'
prompt_reassign += 'Requirement 2: Consider the previous agreement score for the title, which was {}.\n'
prompt_reassign += 'Requirement 3: The new score should fall within the range of {} to {}.\n'
prompt_reassign += 'Requirement 4: The score should be between 0 and 100 and not restricted to multiples of 5 and strictly a single integer number.\n'
prompt_reassign += 'Requirement 5: The output format is [int].'

prompt2 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people believe the content in the title.\n"
prompt2 += 'Requirement 1: The title content is {}.\n'
prompt2 += 'Requirement 2: Please agree with the title content in combination with the following four aspects:\n'
prompt2 += '1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n'
prompt2 += '2. Logic: Are there any leaps in reasoning or inconsistencies?\n'
prompt2 += '3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n'
prompt2 += '4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n'
prompt2 += 'Requirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\n'
prompt2 += 'Requirement 4: The output format is [reasoning content].\n'

prompt3 = "Goal: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity. The inference should make people disbelieve the content in the title.\n"
prompt3 += 'Requirement 1: The title content is {}.\n'
prompt3 += 'Requirement 2: Please disagree with the title content in combination with the following four aspects:\n'
prompt3 += '1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n'
prompt3 += '2. Logic: Are there any leaps in reasoning or inconsistencies?\n'
prompt3 += '3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n'
prompt3 += '4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n'
prompt3 += 'Requirement 3: The length of the reasoning should be limited to 40-60 words, and the content should be placed in [].\n'
prompt3 += 'Requirement 4: The output format is [reasoning content].\n'

prompt4 = 'Goal:  Re-score based on the title content, initial score, and {}reasoning.'
prompt4 += 'Requirement 1: The title is {}.\n'
prompt4 += 'Requirement 2: The initial score is {}.\n'
prompt4 += 'Requirement 3: The {} reasoning content is {}.\n'
prompt4 += 'Requirement 4: The score should be between 0 and 100 and not restricted to multiples of 5 and strictly a single integer number.\n'
prompt4 += 'Requirement 5: The output format is [int].\n'

prompt5 = 'Goal: Analyze the {} reasoning content from the perspectives of rationality and logic.\n'
prompt5 += 'Requirement 1: Consider the previous {} reasoning content: {}\n'
prompt5 += 'Requirement 2: Consider the previous score based on the {} reasoning: {}.\n'
prompt5 += 'Requirement 3: The analysis should be limited to 50-70 words.\n'
prompt5 += 'Requirement 4: Output format [reasoning content].\n'

prompt6 = "Goal: Regenerate {} reasoning content, because the previous reasoning did not effectively {} the title's recognition score\n"
prompt6 += 'Requirement 1: The title is {}.\n'
prompt6 += 'Requirement 2: The initial score is {}.\n'
prompt6 += 'Requirement 3: Consider the previous {} reasoning content: {}.\n'
prompt6 += 'Requirement 4: Consider the title score based on the previous {} reasoning: {}.\n'
prompt6 += 'Requirement 5: Consider the evaluation of the reasoning for {}: {}. \n'
prompt6 += 'Requirement 6: Analyze the logical inconsistencies in the previous reasoning and explain why the new reasoning is more suitable for the title content.\n'
prompt6 += "Requirement 7: New inference generation should combine the following four aspects and adapt to the content of the title to {} people's identification with the content of the title and make people {} in the content of the title."
prompt6 += '1. Common Sense: Does it contain information that is inconsistent with common sense or is obviously wrong?\n'
prompt6 += '2. Logic: Are there any leaps in reasoning or inconsistencies?\n'
prompt6 += '3. Content Completeness: Is there any information that is vague, intentionally left blank, or creates unnecessary suspense?\n'
prompt6 += '4. Objectivity: Is there any judgement, emotional manipulation or inflammatory language?\n'
prompt6 += 'Requirement 8: The limit for inference is 40-60 words, and the limit for explanation is 20-40 words. The inference content is placed in [] and the explanation content is placed in ().\n'
prompt6 += 'Requirement 9: Output format is [Reasoning Content] (Explanatory Content).\n'
prompt6 += 'Requirement 10: The score should still range from 0 to 100, and it should be more humanized, not restricted to multiples of 5 and strictly a single integer number.\n'
prompt6 += 'Requirement 11: Output format for the score is [int].\n'

# =================================================================================
# PHẦN 4: LOGIC XỬ LÝ 1 DÒNG DỮ LIỆU (CORE BUSINESS LOGIC)
# =================================================================================

def process_single_row(row_data, client):
    """
    Xử lý một dòng dữ liệu - BẢO TOÀN 100% LOGIC NGHIỆP VỤ

    Args:
        row_data: Dictionary chứa thông tin 1 dòng (index, title, subtitle, label, etc.)
        client: OpenAI client để gọi API

    Returns:
        Dictionary chứa kết quả xử lý
    """
    index = row_data['index']
    title = row_data['title']
    subtitle = row_data.get('subtitle', None)

    # Xây dựng input_text
    if subtitle is None or pd.isna(subtitle):
        input_text = str(title)
    else:
        input_text = str(title) + " " + str(subtitle)

    # Các chuỗi constant - giữ nguyên từ SORG_1.py
    str3 = 'agree'
    str4 = 'disagree'
    str5 = 'increase'
    str6 = 'lower'
    str7 = 'believe'
    str8 = 'disbelieve'

    agree_reason_all = ""
    ret_agree_reason_all = ""
    disagree_reason_all = ""

    # =========================================================================
    # BƯỚC 1: Tính original_score - GIỮ NGUYÊN LOGIC while original_score < 30 or > 70
    # =========================================================================
    original_score = prompt1.format(input_text)
    original_score = generate_score(original_score, client)

    while original_score < 30 or original_score > 70:
        original_score = prompt_reassign.format(input_text, original_score, 30, 70)
        original_score = generate_score(original_score, client)

    # =========================================================================
    # BƯỚC 2: Tạo agree_reason và agr_score - GIỮ NGUYÊN LOGIC while với count1
    # =========================================================================
    agree_reason = prompt2.format(input_text)
    agree_reason = generate_res(agree_reason, client)
    agree_reason = extract_quoted_text(agree_reason, input_text)
    agree_reason_all = agree_reason

    q1 = prompt4.format(str3, input_text, original_score, str3, agree_reason)
    agr_score = generate_score(q1, client)
    agr_score_all = agr_score

    agree_reason = [agree_reason]
    count1 = 0

    # Vòng lặp while cho agree - GIỮ NGUYÊN ĐIỀU KIỆN
    while agr_score - original_score < 10 or agr_score <= 55:
        print(f"[Index {index}] ********对认同推理内容分析******** (count1={count1})")

        ret_agree_reason = prompt5.format(str3, str3, agree_reason, str3, agr_score)
        ret_agree_reason = generate_res(ret_agree_reason, client)
        ret_agree_reason_all += f"$$$$$ {ret_agree_reason}"

        print(f"[Index {index}] ********重新生成认同推理********")
        agree_reason = prompt6.format(str3, str5, input_text, original_score, str3, agree_reason, str3, agr_score, str3, ret_agree_reason, str5, str7)
        agree_reason = generate_res(agree_reason, client)
        agree_reason = extract_quoted_text(agree_reason, input_text)
        agree_reason_all += f"$$$$$ {agree_reason}"

        q1 = prompt6.format(str3, str5, input_text, original_score, str3, agree_reason, str3, agr_score, str3, ret_agree_reason, str5, str7)
        agr_score = generate_score(q1, client)
        agr_score_all = str(agr_score_all) + f"$$ {agr_score}"

        count1 += 1
        if count1 == 20:
            break

    # =========================================================================
    # BƯỚC 3: Tạo disagree_reason và dis_score - GIỮ NGUYÊN LOGIC while với count2
    # =========================================================================
    disagree_reason = prompt3.format(input_text)
    disagree_reason = generate_res(disagree_reason, client)
    disagree_reason = extract_quoted_text(disagree_reason, input_text)
    disagree_reason_all = disagree_reason

    q_neg = prompt4.format(str4, input_text, original_score, str4, disagree_reason)
    dis_score = generate_score(q_neg, client)
    dis_score_all = dis_score

    disagree_reason = [disagree_reason]
    count2 = 0

    # Vòng lặp while cho disagree - GIỮ NGUYÊN ĐIỀU KIỆN
    while original_score - dis_score < 10 or dis_score >= 45:
        print(f"[Index {index}] ********Content Analysis of Disagreement Reasoning******** (count2={count2})")

        ret_disagree_reason = prompt5.format(str4, str4, disagree_reason, str4, dis_score)
        ret_disagree_reason = generate_res(ret_disagree_reason, client)

        print(f"[Index {index}] ********Regenerate Disagreement Reasoning********")
        disagree_reason = prompt6.format(str4, str6, input_text, original_score, str4, disagree_reason, str4, dis_score, str4, ret_disagree_reason, str6, str8)
        disagree_reason = generate_res(disagree_reason, client)
        disagree_reason = extract_quoted_text(disagree_reason, input_text)
        disagree_reason_all += f"$$$$$ {disagree_reason}"

        q_neg = prompt6.format(str4, str6, input_text, original_score, str4, disagree_reason, str4, dis_score, str4, ret_disagree_reason, str6, str8)
        dis_score = generate_score(q_neg, client)
        dis_score_all = str(dis_score_all) + f"$$ {dis_score}"

        count2 += 1
        if count2 == 20:
            break

    # =========================================================================
    # Trả về kết quả
    # =========================================================================
    result = {
        'index': index,
        'agree_reason': str(agree_reason),
        'disagree_reason': str(disagree_reason),
        'agree_reason_all': str(agree_reason_all),
        'disagree_reason_all': str(disagree_reason_all),
        'original_score': original_score,
        'agree_score': agr_score,
        'disagree_score': dis_score,
        'agree_score_all': str(agr_score_all),
        'disagree_score_all': str(dis_score_all)
    }

    return result

# =================================================================================
# PHẦN 5: THREADING - XỬ LÝ ĐỒNG THỜI TRONG MỘT PROCESS
# =================================================================================

def process_batch_with_threading(batch_df, max_workers=5):
    """
    Xử lý một batch dữ liệu với ThreadPoolExecutor

    Args:
        batch_df: DataFrame chứa batch cần xử lý
        max_workers: Số lượng thread đồng thời (MAX_CONCURRENT=5)

    Returns:
        List of dictionaries chứa kết quả
    """
    # Tạo OpenAI client cho batch này (mỗi thread sẽ dùng chung client này)
    client = create_openai_client()

    results = []

    # Chuyển DataFrame thành list of dictionaries
    rows_data = []
    for idx, row in batch_df.iterrows():
        row_dict = row.to_dict()
        row_dict['index'] = idx
        rows_data.append(row_dict)

    # Sử dụng ThreadPoolExecutor để xử lý đồng thời MAX_CONCURRENT=5
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tất cả các tasks
        future_to_row = {
            executor.submit(process_single_row, row_data, client): row_data
            for row_data in rows_data
        }

        # Thu thập kết quả khi hoàn thành
        for future in as_completed(future_to_row):
            row_data = future_to_row[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing index {row_data['index']}: {e}")

    return results

# =================================================================================
# PHẦN 6: WORKER FUNCTION CHO MULTIPROCESSING
# =================================================================================

def worker_process(process_id, data_chunk, chunk_size=100, max_concurrent=5):
    """
    Hàm xử lý cho mỗi process (MULTIPROCESSING)

    Args:
        process_id: ID của process (để tracking)
        data_chunk: DataFrame chunk được giao cho process này
        chunk_size: Kích thước mỗi batch nhỏ (CHUNK_SIZE=100)
        max_concurrent: Số lượng thread đồng thời trong process (MAX_CONCURRENT=5)

    Returns:
        List of dictionaries chứa toàn bộ kết quả của process này
    """
    print(f"[Process {process_id}] Bắt đầu xử lý {len(data_chunk)} dòng dữ liệu")

    all_results = []

    # Chia data_chunk thành các batch nhỏ hơn (CHUNK_SIZE=100)
    num_batches = int(np.ceil(len(data_chunk) / chunk_size))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * chunk_size
        end_idx = min((batch_idx + 1) * chunk_size, len(data_chunk))

        batch_df = data_chunk.iloc[start_idx:end_idx]

        print(f"[Process {process_id}] Đang xử lý batch {batch_idx + 1}/{num_batches} "
              f"(dòng {start_idx} đến {end_idx})")

        # Xử lý batch này với Threading (MAX_CONCURRENT=5)
        batch_results = process_batch_with_threading(batch_df, max_workers=max_concurrent)
        all_results.extend(batch_results)

    print(f"[Process {process_id}] Hoàn thành xử lý {len(all_results)} dòng")
    return all_results

# =================================================================================
# PHẦN 7: PARSE CLI ARGUMENTS
# =================================================================================

def parse_arguments():
    """
    Parse command-line arguments với tên đầy đủ và tên viết tắt

    Returns:
        argparse.Namespace: Object chứa tất cả arguments
    """
    parser = argparse.ArgumentParser(
        description='Hệ thống phân loại Clickbait với Multiprocessing + Threading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==================== API Configuration ====================
    parser.add_argument('-d', '--data-path',
                        type=str,
                        default='clickbait_data_1.csv',
                        help='Đường dẫn đến file CSV đầu vào')

    parser.add_argument('-o', '--output',
                        type=str,
                        default='sorg_final_output.csv',
                        help='Đường dẫn đến file CSV đầu ra')

    parser.add_argument('-m', '--model',
                        type=str,
                        default='gpt-4o-mini-2024-07-18',
                        help='Tên model OpenAI cần sử dụng')

    parser.add_argument('-k', '--api-key',
                        type=str,
                        default='sk-HlkHAJijdlayeUE5y2pmyyIJgjZEK5Tzutrb8mvz29BWhDiE',
                        help='API key cho OpenAI')

    parser.add_argument('-u', '--base-url',
                        type=str,
                        default='https://api-v2.shopaikey.com/v1',
                        help='Base URL của API OpenAI')

    # ==================== Multiprocessing Configuration ====================
    parser.add_argument('-p', '--num-processes',
                        type=int,
                        default=None,
                        help='Số lượng processes song song (None = tự động = số CPU cores)')

    parser.add_argument('-c', '--chunk-size',
                        type=int,
                        default=100,
                        help='Kích thước mỗi batch trong process (số dòng)')

    parser.add_argument('-w', '--max-concurrent',
                        type=int,
                        default=5,
                        help='Số lượng API calls đồng thời trong 1 process')

    # ==================== API Parameters - Score ====================
    parser.add_argument('-ts', '--temp-score',
                        type=float,
                        default=0.1,
                        help='Temperature cho API calls tính điểm (score)')

    parser.add_argument('-ps', '--top-p-score',
                        type=float,
                        default=1.0,
                        help='Top-p (nucleus sampling) cho API calls tính điểm')

    parser.add_argument('-ms', '--max-tokens-score',
                        type=int,
                        default=100,
                        help='Max tokens cho API calls tính điểm')

    # ==================== API Parameters - Reasoning ====================
    parser.add_argument('-tr', '--temp-reasoning',
                        type=float,
                        default=0.3,
                        help='Temperature cho API calls reasoning')

    parser.add_argument('-pr', '--top-p-reasoning',
                        type=float,
                        default=1.0,
                        help='Top-p (nucleus sampling) cho API calls reasoning')

    parser.add_argument('-mr', '--max-tokens-reasoning',
                        type=int,
                        default=150,
                        help='Max tokens cho API calls reasoning')

    # ==================== Retry & Other ====================
    parser.add_argument('-rt', '--max-retries',
                        type=int,
                        default=5,
                        help='Số lần retry tối đa khi API call thất bại')

    parser.add_argument('-r', '--random-state',
                        type=int,
                        default=42,
                        help='Random state cho việc shuffle dataset')

    args = parser.parse_args()
    return args

# =================================================================================
# PHẦN 8: MAIN FUNCTION - ĐIỀU PHỐI MULTIPROCESSING
# =================================================================================

def main():
    """
    Hàm chính - Điều phối toàn bộ quá trình Multiprocessing
    Parse CLI arguments và cập nhật CONFIG trước khi xử lý
    """
    # Parse CLI arguments
    args = parse_arguments()

    # Cập nhật CONFIG từ CLI arguments
    CONFIG['base_url'] = args.base_url
    CONFIG['api_key'] = args.api_key
    CONFIG['model'] = args.model
    CONFIG['max_retries'] = args.max_retries
    CONFIG['temp_score'] = args.temp_score
    CONFIG['temp_reasoning'] = args.temp_reasoning
    CONFIG['top_p_score'] = args.top_p_score
    CONFIG['top_p_reasoning'] = args.top_p_reasoning
    CONFIG['max_tokens_score'] = args.max_tokens_score
    CONFIG['max_tokens_reasoning'] = args.max_tokens_reasoning

    print("="*80)
    print("BẮT ĐẦU QUÁ TRÌNH XỬ LÝ VỚI MULTIPROCESSING + THREADING")
    print("="*80)

    # In ra cấu hình
    print("\n[CẤU HÌNH]")
    print(f"  Data Path: {args.data_path}")
    print(f"  Output Path: {args.output}")
    print(f"  Model: {args.model}")
    print(f"  Num Processes: {args.num_processes or 'Auto (CPU count)'}")
    print(f"  Chunk Size: {args.chunk_size}")
    print(f"  Max Concurrent: {args.max_concurrent}")
    print(f"  Max Retries: {args.max_retries}")
    print(f"  Temperature (Score): {args.temp_score}")
    print(f"  Temperature (Reasoning): {args.temp_reasoning}")
    print(f"  Random State: {args.random_state}")

    # =========================================================================
    # Đọc và xáo trộn dataset
    # =========================================================================
    print(f"\n[BƯỚC 1] Đọc dataset từ {args.data_path}...")
    try:
        unshuff_data = pd.read_csv(args.data_path)
    except FileNotFoundError:
        print(f"[LỖI] Không tìm thấy file: {args.data_path}")
        return
    except Exception as e:
        print(f"[LỖI] Lỗi đọc file: {e}")
        return

    data = unshuff_data.sample(frac=1, random_state=args.random_state)  # Xáo trộn
    print(f"Tổng số dòng: {len(data)}")

    # Thêm các cột kết quả
    data['agree_reason'] = None
    data['disagree_reason'] = None
    data['agree_reason_all'] = None
    data['disagree_reason_all'] = None
    data['original_score'] = None
    data['agree_score'] = None
    data['disagree_score'] = None
    data['agree_score_all'] = None
    data['disagree_score_all'] = None

    # =========================================================================
    # Chia dataset thành N phần cho N processes
    # =========================================================================
    num_processes = args.num_processes if args.num_processes else mp.cpu_count()
    print(f"\n[BƯỚC 2] Chia dataset thành {num_processes} phần cho {num_processes} processes...")

    # Chia đều dataset
    chunk_size_per_process = int(np.ceil(len(data) / num_processes))
    data_chunks = []

    for i in range(num_processes):
        start_idx = i * chunk_size_per_process
        end_idx = min((i + 1) * chunk_size_per_process, len(data))
        chunk = data.iloc[start_idx:end_idx].copy()
        data_chunks.append(chunk)
        print(f"  Process {i}: {len(chunk)} dòng (từ {start_idx} đến {end_idx})")

    # =========================================================================
    # Khởi động Multiprocessing Pool
    # =========================================================================
    print(f"\n[BƯỚC 3] Khởi động {num_processes} processes với Pool...")

    with mp.Pool(processes=num_processes) as pool:
        # Tạo arguments cho mỗi worker
        tasks = [
            (process_id, chunk, args.chunk_size, args.max_concurrent)
            for process_id, chunk in enumerate(data_chunks)
        ]

        # Chạy song song - mỗi process xử lý chunk của mình
        results_per_process = pool.starmap(worker_process, tasks)

    # =========================================================================
    # Gộp kết quả từ tất cả các processes
    # =========================================================================
    print("\n[BƯỚC 4] Gộp kết quả từ tất cả các processes...")
    all_results = []
    for process_results in results_per_process:
        all_results.extend(process_results)

    print(f"Tổng số dòng đã xử lý: {len(all_results)}")

    # =========================================================================
    # Cập nhật DataFrame với kết quả
    # =========================================================================
    print("\n[BƯỚC 5] Cập nhật DataFrame với kết quả...")
    for result in all_results:
        idx = result['index']
        data.at[idx, 'agree_reason'] = result['agree_reason']
        data.at[idx, 'disagree_reason'] = result['disagree_reason']
        data.at[idx, 'agree_reason_all'] = result['agree_reason_all']
        data.at[idx, 'disagree_reason_all'] = result['disagree_reason_all']
        data.at[idx, 'original_score'] = result['original_score']
        data.at[idx, 'agree_score'] = result['agree_score']
        data.at[idx, 'disagree_score'] = result['disagree_score']
        data.at[idx, 'agree_score_all'] = result['agree_score_all']
        data.at[idx, 'disagree_score_all'] = result['disagree_score_all']

    # =========================================================================
    # Lưu kết quả ra file DUY NHẤT
    # =========================================================================
    print(f"\n[BƯỚC 6] Lưu kết quả ra file {args.output}...")
    try:
        data.to_csv(args.output, index=False)
        print("\n" + "="*80)
        print("HOÀN THÀNH! Kết quả đã được lưu vào:", args.output)
        print("="*80)
    except Exception as e:
        print(f"[LỖI] Không thể lưu file: {e}")

# =================================================================================
# ENTRY POINT
# =================================================================================

if __name__ == '__main__':
    # Cần thiết cho multiprocessing trên Windows
    mp.freeze_support()
    main()
