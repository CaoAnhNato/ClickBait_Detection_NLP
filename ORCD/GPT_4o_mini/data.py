import pandas as pd

# Khởi tạo một list rỗng để chứa dữ liệu
data = []

# Đọc file 'clickbait' và gán nhãn 1
with open('clickbait-detection/clickbait', 'r', encoding='utf-8') as f1:
    for line in f1:
        title = line.strip()
        if title:  # Kiểm tra để bỏ qua các dòng trống (nếu có)
            data.append({'title': title, 'label': 1})

# Đọc file 'not-clickbait' và gán nhãn 0
with open('clickbait-detection/not-clickbait', 'r', encoding='utf-8') as f0:
    for line in f0:
        title = line.strip()
        if title:
            data.append({'title': title, 'label': 0})

# Tạo DataFrame từ list data
df = pd.DataFrame(data)

# Trộn ngẫu nhiên các hàng để dữ liệu không bị phân dồn theo file (Tùy chọn)
df = df.sample(frac=1).reset_index(drop=True)

# Hiển thị thử 5 dòng đầu tiên
print(df.head())

df.to_csv('clickbait_data.csv', index=False)