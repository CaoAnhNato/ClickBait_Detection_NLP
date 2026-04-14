import pandas as pd

# Thay đường dẫn này thành đường dẫn file .tsv thực tế của bạn
file_path = 'muser_gossipcop_content_no_ignore.tsv'

# Đọc file .tsv vào DataFrame
df = pd.read_csv(file_path, sep='\t')

# Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu
print(df.head())
print(df['label'].value_counts())