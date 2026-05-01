from openai import OpenAI

# Khởi tạo client
client = OpenAI(
    api_key='sk-RcgPIw5nCUPMPRBydQNn7cWb68ESkyEaOoPCsT6s5IdaqX3v',
    base_url='https://api-v2.shopaikey.com/v1',
)

# Gọi API tạo câu trả lời
completion = client.chat.completions.create(
    model='gpt-4o',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Xin chào!'}
    ],
    max_tokens=1000,
    temperature=1.0,
)

# In kết quả ra màn hình
print(completion.choices[0].message.content)