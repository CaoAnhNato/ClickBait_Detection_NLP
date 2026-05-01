import time
import asyncio
import json
from openai import AsyncOpenAI

# =================================================================================
# QUICK CONFIGURATION
# =================================================================================
API_KEY = "sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v"
BASE_URL = "https://api-v2.shopaikey.com/v1"
MODEL_NAME = "gpt-3.5-turbo-1106"

def create_async_openai_client():
    return AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

async def process_single_row(title: str, client: AsyncOpenAI) -> dict:
    prompt = f"""
Goal: As a news expert, evaluate the title's content and score it according to the criteria below.
Requirement 1: The title is '{title}'.
Requirement 2: Make a comprehensive inference about the title from four aspects: common sense, logic, content integrity, and objectivity.
Requirement 3: First, assign an "original_score" representing the general public's agreement/belief level with the title (30 to 70).
Requirement 4: Then, mentally formulate an "agree_reason" (40-60 words) that advocates for the title being completely truthful. Based on this, assign an "agree_score" that must be at least 15 points higher than original_score (up to 100).
Requirement 5: Finally, mentally formulate a "disagree_reason" (40-60 words) that highlights any illogical leaps or vague language. Based on this, assign a "disagree_score" that must be at least 15 points lower than original_score (down to 0).
Requirement 6: All scores should be formatted as strictly single integers.
Requirement 7: The output MUST be a valid JSON object with EXACTLY three numeric fields: "original_score", "agree_score", "disagree_score". Do not output the reason text, just the final scores.
"""
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        response_format={ "type": "json_object" },
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )
    res_content = response.choices[0].message.content
    try:
        data = json.loads(res_content)
        return {
            'original_score': int(data.get('original_score', 50)),
            'agree_score': int(data.get('agree_score', 80)),
            'disagree_score': int(data.get('disagree_score', 20)),
        }
    except Exception as e:
        return {'original_score': 50, 'agree_score': 85, 'disagree_score': 35}

async def main():
    print("=== FAST PIPELINE SORG DATA GENERATION (< 6s/Title) ===")
    client = create_async_openai_client()
    
    test_cases = [
        "This Vine Of New York On \"Celebrity Big Brother\" Is Fucking Perfect",
        "17 Hairdresser Struggles Every Black Girl Knows To Be True",
        "Here's One Really Weird Thing About Butterfree",
        "Coldplay's new album hits stores worldwide this week",
        "New law to help asbestos sufferers in Victoria, Australia",
        "Car Bomb Kills Police Official in Spain"
    ]
    
    total_start_time = time.time()
    
    for i, title in enumerate(test_cases, 1):
        print(f"\n[+] Bắt đầu xử lý title {i}/6: '{title}'")
        start_time = time.perf_counter()
        
        result = await process_single_row(title, client)
        elapsed_time = time.perf_counter() - start_time
        
        print(f"    => Original: {result['original_score']} | Agree: {result['agree_score']} | Disagree: {result['disagree_score']}")
        print(f"    ⏱ Thời gian thực thi mẫu này: {elapsed_time:.2f} giây")
        if elapsed_time > 6.0:
            print(f"    [WARNING] Mẫu này vượt quá 6s!")

    print("-" * 50)
    print(f"TỔNG THỜI GIAN 6 CASES (TUẦN TỰ): {time.time() - total_start_time:.2f} giây")

if __name__ == "__main__":
    asyncio.run(main())