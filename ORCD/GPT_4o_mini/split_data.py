import pandas as pd
import numpy as np

# Đọc file dữ liệu gốc
file_path = 'clickbait_data.csv'
data = pd.read_csv(file_path)

# Trộn đều dữ liệu (shuffle) để đảm bảo phân bổ đều clickbait/non-clickbait
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Chia thành 6 phần (chunks)
chunks = np.array_split(data, 10)

# Lưu thành 6 file CSV riêng biệt
for i, chunk in enumerate(chunks):
    output_name = f'clickbait_data_{i+1}.csv'
    chunk.to_csv(output_name, index=False)
    print(f"Đã lưu {output_name} với {len(chunk)} mẫu.")