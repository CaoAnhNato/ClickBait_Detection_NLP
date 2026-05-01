import os
from openai import OpenAI

# Khởi tạo client với API key của bạn
# Đừng quên thay "YOUR_API_KEY" bằng key thực tế của bạn nhé.
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"), 
    base_url="https://api-v2.shopaikey.com/v1"
)

def ask_gpt4o(question):
    try:
        # Gọi API tạo phản hồi (Chat Completion)
        response = client.chat.completions.create(
            model="gpt-4o", # Chỉ định model gpt-4o
            messages=[
                {"role": "system", "content": "Bạn là một trợ lý AI trả lời ngắn gọn, súc tích."},
                {"role": "user", "content": question}
            ]
        )
        # Trích xuất nội dung câu trả lời
        return response.choices[0].message.content
    except Exception as e:
        return f"Đã có lỗi xảy ra: {e}"

# Đặt một câu hỏi đơn giản
cau_hoi = "Khoảng cách từ Trái Đất đến Mặt Trăng là bao nhiêu?"
cau_tra_loi = ask_gpt4o(cau_hoi)

print(f"Hỏi: {cau_hoi}")
print(f"GPT-4o: {cau_tra_loi}")