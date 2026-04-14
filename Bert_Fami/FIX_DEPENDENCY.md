# Hướng Dẫn Fix Lỗi Dependency - RTX 5080 Training

## 🔴 Vấn Đề Hiện Tại

Bạn đang gặp **2 vấn đề nghiêm trọng về compatibility**:

### 1. Transformers Library Incompatibility
```
Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.1
```
- **Nguyên nhân**: Phiên bản `transformers` quá mới (>= 4.46) yêu cầu PyTorch >= 2.4
- **Hiện tại**: Bạn có PyTorch 2.2.1

### 2. GPU Architecture Incompatibility
```
NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible
```
- **Nguyên nhân**: RTX 5080 (Blackwell architecture, sm_120) chưa được PyTorch 2.2.1 hỗ trợ
- **Hỗ trợ**: PyTorch 2.2.1 chỉ support: sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90

---

## ✅ GIẢI PHÁP 1: Nâng Cấp PyTorch (KHUYẾN NGHỊ)

**Đây là giải pháp TỐT NHẤT** để tận dụng RTX 5080 16GB của bạn.

### Bước 1: Gỡ cài đặt PyTorch cũ
```bash
pip uninstall torch torchvision torchaudio -y
```

### Bước 2: Cài đặt PyTorch 2.5.1 (hỗ trợ RTX 5080)
```bash
# CUDA 12.1
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hoặc CUDA 12.4
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Bước 3: Kiểm tra cài đặt
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Kết quả mong đợi:
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 5080
```

### Bước 4: Chạy training
```bash
python3 train_bert.py -d clickbait_data.csv -b 24 -nw 6 --use-amp --compile-model -ga 2
```

---

## ✅ GIẢI PHÁP 2: Hạ Cấp Transformers (Tạm thời)

**Chỉ dùng khi không thể nâng cấp PyTorch**, nhưng **KHÔNG THỂ dùng GPU RTX 5080**.

### Bước 1: Hạ cấp transformers
```bash
pip install transformers==4.45.2
```

### Bước 2: Chạy training trên CPU
```bash
python3 train_bert.py -d clickbait_data.csv -b 8 -nw 4
```

**Lưu ý**: Training trên CPU sẽ **rất chậm** (10-50x chậm hơn GPU).

---

## ✅ GIẢI PHÁP 3: Cài Đặt PyTorch Nightly (Experimental)

Nếu PyTorch 2.5.1 stable vẫn chưa support RTX 5080:

```bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

---

## 🔍 Kiểm Tra Phiên Bản Hiện Tại

```bash
# Kiểm tra PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Kiểm tra Transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Kiểm tra CUDA
python3 -c "import torch; print(f'CUDA: {torch.version.cuda}')"

# Kiểm tra GPU compatibility
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 Bảng Tương Thích

| PyTorch Version | Transformers Version | RTX 5080 Support | Status |
|----------------|---------------------|------------------|---------|
| 2.2.1 | >= 4.46 | ❌ | ❌ Không tương thích |
| 2.2.1 | < 4.46 | ❌ | ⚠️ CPU only |
| 2.5.1+ | Any | ✅ | ✅ **Khuyến nghị** |

---

## 🚀 Sau Khi Fix Xong

### Training với full GPU optimization:
```bash
python3 train_bert.py \
    -d clickbait_data.csv \
    -b 32 \
    -nw 8 \
    --use-amp \
    --compile-model \
    -ga 2
```

### Kiểm tra GPU usage trong lúc training:
```bash
# Terminal khác
watch -n 1 nvidia-smi
```

---

## 📝 File Code Đã Được Cập Nhật

File `train_bert.py` đã được thêm:
1. ✅ **Automatic compatibility checking** - Tự động kiểm tra version conflicts
2. ✅ **Clear error messages** - Thông báo lỗi rõ ràng với hướng dẫn fix
3. ✅ **CPU fallback mode** - Tự động chuyển sang CPU nếu GPU không tương thích
4. ✅ **Graceful exit** - Thoát an toàn với hướng dẫn

---

## ❓ FAQ

**Q: Tại sao không dùng transformers mới nhất với PyTorch 2.2.1?**
A: Transformers >= 4.46 sử dụng các API chỉ có trong PyTorch 2.4+. Không thể workaround.

**Q: Có cách nào dùng RTX 5080 với PyTorch 2.2.1?**
A: Không. RTX 5080 (Blackwell, sm_120) cần PyTorch 2.5+ với driver support.

**Q: Nên chọn CUDA 12.1 hay 12.4?**
A: Kiểm tra CUDA driver của bạn với `nvidia-smi`. Chọn phiên bản tương ứng.

**Q: Training trên CPU có khả thi không?**
A: Có thể, nhưng rất chậm. Dataset nhỏ (~2k samples) có thể mất 10-20 giờ.

---

## 📞 Support

Nếu vẫn gặp vấn đề sau khi làm theo hướng dẫn:
1. Kiểm tra lại tất cả các bước
2. Gửi output của các lệnh kiểm tra phiên bản
3. Gửi full error log
