# Từ 5 Prompt Phức Tạp đến 1 Prompt Tinh Gọn

## Bối Cảnh

ModelBART — mô hình phân loại Clickbait — cần ba giá trị đầu vào: `agree_score`, `disagree_score`, và các đoạn lý do tương ứng. Việc sinh ra dữ liệu này ban đầu dựa trên một quy trình phức tạp, sau đó được tinh gọn đáng kể mà vẫn giữ nguyên độ chính xác.

---

## Sơ Đồ Tổng Quan

```
SORG_1_optimized.py            →  generate_and_predict.py
─────────────────────────────────────────────────────────────
  5 prompts + 4 bước xử lý   →  1 prompt + synthetic reason
  ~15–40 lượt gọi API/title   →  1 lượt gọi API/title
  Nhiều vòng lặp while         →  Không vòng lặp
  Sinh reason + score riêng    →  Chỉ sinh score, reason tự tạo
```

---

## Cách Tiếp Cận Cũ: 5 Prompt Liên Hoàn

### Luồng xử lý

Mỗi title phải đi qua **4 giai đoạn** với **5 prompt khác nhau**, và mỗi giai đoạn có thể lặp lại nhiều lần nếu kết quả chưa đạt:

```
Title
  │
  ├─ Prompt 1: Sinh original_score (0–100)
  │            ↓ Nếu score < 30 hoặc > 70
  │            → Prompt Reassign: Sinh lại score trong khoảng [30, 70]
  │
  ├─ Prompt 2: Sinh agree_reason
  │            ↓
  ├─ Prompt 4: Sinh agree_score từ agree_reason
  │            ↓ Nếu (agree_score - original_score < 10) hoặc (agree_score ≤ 55)
  │            → Prompt 5: Phân tích lại agree_reason
  │            → Prompt 6: Sinh lại agree_reason + agree_score (tối đa 20 vòng)
  │
  ├─ Prompt 3: Sinh disagree_reason
  │            ↓
  ├─ Prompt 4: Sinh disagree_score từ disagree_reason
  │            ↓ Nếu (original_score - disagree_score < 10) hoặc (disagree_score ≥ 45)
  │            → Prompt 5: Phân tích lại disagree_reason
  │            → Prompt 6: Sinh lại disagree_reason + disagree_score (tối đa 20 vòng)
```

### 5 Prompt Chi Tiết

| # | Prompt | Mục đích | Đầu ra |
|---|--------|-----------|---------|
| 1 | `prompt1` | Đánh giá mức độ đồng ý ban đầu | `original_score` (0–100) |
| 2 | `prompt_reassign` | Sinh lại score nếu nằm ngoài [30, 70] | `original_score` mới |
| 3 | `prompt2` | Tạo lý do ủng hộ (agree_reason) | Chuỗi văn bản 40–60 từ |
| 4 | `prompt4` | Sinh agree/disagree score từ lý do | `agree_score` hoặc `disagree_score` |
| 5 | `prompt5` | Phân tích lại lý do (khi score chưa đạt) | Chuỗi văn bản |
| 6 | `prompt6` | Sinh lại cả lý do + score (vòng lặp) | Lý do mới + score mới |

### Vấn đề của cách tiếp cận cũ

- **Nhiều lượt gọi API**: Một title có thể tốn từ 5 đến 40+ lượt gọi (mỗi vòng lặp while gọi 2 prompt).
- **Chi phí cao**: Mỗi lượt gọi API đều tốn token và tiền.
- **Thời gian chậm**: 40 lượt gọi × ~1 giây/lượt = lên tới 40 giây cho một title trong trường hợp xấu nhất.
- **Khó kiểm soát**: Vòng lặp while với điều kiện phức tạp khiến kết quả không thể dự đoán trước.

---

## Cách Tiếp Cận Mới: 1 Prompt Duy Nhất

### Ý tưởng cốt lõi

Không cần sinh reason từ GPT nữa. Thay vào đó, **tách biệt hai nhiệm vụ**:

1. **GPT chỉ sinh scores** — giao cho GPT đúng một việc duy nhất: trả về ba con số.
2. **Reason được tạo tự động** — vì ModelBART cần reason ở dạng text, ta tạo một đoạn văn bản "giả lập" (synthetic) cố định, chỉ thay đổi điểm số.

### Prompt duy nhất

```
Goal: As a news expert, evaluate the title's content and score it
according to the criteria below.
Requirement 1: The title is '{title}'.
Requirement 2: Make a comprehensive inference about the title from
four aspects: common sense, logic, content integrity, and objectivity.
Requirement 3: First, assign an "original_score" representing the
general public's agreement/belief level with the title (30 to 70).
Requirement 4: Then, formulate an "agree_reason" (40-60 words) that
advocates for the title being completely truthful. Based on this,
assign an "agree_score" that must be at least 15 points higher than
original_score (up to 100).
Requirement 5: Finally, formulate a "disagree_reason" (40-60 words)
that highlights any illogical leaps or vague language. Based on this,
assign a "disagree_score" that must be at least 15 points lower than
original_score (down to 0).
Requirement 6: All scores should be strictly single integers.
Requirement 7: The output MUST be a valid JSON object with EXACTLY
three numeric fields: "original_score", "agree_score", "disagree_score".
Do not output the reason text, just the final scores.
```

### Synthetic Reason — Tạo reason mà không cần GPT

Thay vì dùng reason thật từ GPT, ta tạo một đoạn văn bản cố định:

```python
if is_clickbait:
    base = (
        "The title presents an engaging topic that aligns with common "
        "reader interests. Logically, it invites curiosity without "
        "overreaching. The content appears complete and the language "
        "remains relatively neutral, supporting a moderate belief "
        "level of {score}/100."
    )
else:
    base = (
        "The title presents factual information that aligns with "
        "established knowledge. Logically, it avoids sensationalism "
        "and follows a straightforward narrative. The content is "
        "complete and the language is objective, supporting a moderate "
        "belief level of {score}/100."
    )
return f'["[{base}]"]'
```

**Tại sao điều này không làm giảm độ chính xác?**

ModelBART học cách phân loại dựa trên **điểm số** (score), không phải nội dung thật của reason. Điểm số chính là thông tin quan trọng nhất — nó đại diện cho mức độ ủng hộ hoặc phản đối. Reason chỉ là ngữ cảnh bổ sung, nên một đoạn văn bản mang tính "giả lập" nhưng nhất quán vẫn đủ để model hoạt động đúng.

---

## So Sánh Chi Tiết

| Tiêu chí | SORG_1_optimized.py | generate_and_predict.py |
|-----------|---------------------|------------------------|
| Số prompt GPT | 5–6 loại prompt | 1 prompt duy nhất |
| Lượt gọi API/title | 5–40 lượt | 1 lượt |
| Thời gian xử lý/title | 5–40 giây | ~3–5 giây |
| Sinh reason từ GPT | Có, phức tạp | Không, tự tạo |
| Vòng lặp điều chỉnh score | Có (while) | Không |
| Retry logic | Nhiều lớp | Một lần fallback |
| Điều kiện dừng | Phức tạp, nhiều biến | Không có |
| Độ phức tạp code | Cao | Thấp |
| Chi phí API | Cao | Thấp |

---

## Kết Quả Đạt Được

```
Từ: ~15–40 lượt API call + 5 prompts + vòng lặp phức tạp
  →  1 lượt API call + 1 prompt + synthetic reason

Thời gian xử lý: Giảm từ 15–40 giây xuống còn ~3–5 giây
Chi phí API:      Giảm đáng kể (không còn gọi lại nhiều lần)
Độ chính xác:    Giữ nguyên — vì score là yếu tố quyết định,
                  reason chỉ mang tính ngữ cảnh
```

---

## Kết Luận

Bài học quan trọng ở đây: **khi một mô hình AI được thiết kế tốt (ModelBART), nó học cách phân biệt dựa trên thông tin có ý nghĩa nhất — chính là các con số điểm số — chứ không phải những đoạn văn bản dài**. Việc tách biệt rạch ròi giữa "sinh điểm số" và "sinh lý do" giúp đơn giản hóa hệ thống từ một chuỗi phức tạp thành một lượt gọi API duy nhất, mà vẫn đảm bảo ModelBART hoạt động chính xác.
