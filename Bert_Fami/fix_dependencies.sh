#!/bin/bash
# Auto-fix script for PyTorch and Transformers compatibility issues

echo "=================================="
echo "Dependency Fix Script for RTX 5080"
echo "=================================="
echo ""

# Check current versions
echo "Checking current versions..."
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
TRANSFORMERS_VERSION=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not installed")

echo "Current PyTorch: $TORCH_VERSION"
echo "Current Transformers: $TRANSFORMERS_VERSION"
echo ""

# Ask user which solution to apply
echo "Choose a solution:"
echo "1. Upgrade PyTorch to 2.5.1 (RECOMMENDED for RTX 5080)"
echo "2. Downgrade Transformers to 4.45.2 (CPU training only)"
echo "3. Install PyTorch Nightly (Experimental)"
echo "4. Cancel"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Upgrading PyTorch to 2.5.1..."
        echo ""

        # Ask for CUDA version
        echo "Which CUDA version do you have?"
        echo "Check with: nvidia-smi"
        echo "1. CUDA 12.1"
        echo "2. CUDA 12.4"
        read -p "Enter choice (1-2): " cuda_choice

        echo ""
        echo "Uninstalling old PyTorch..."
        pip uninstall torch torchvision torchaudio -y

        if [ "$cuda_choice" = "1" ]; then
            echo "Installing PyTorch 2.5.1 with CUDA 12.1..."
            pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        else
            echo "Installing PyTorch 2.5.1 with CUDA 12.4..."
            pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
        fi

        echo ""
        echo "Verifying installation..."
        python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

        echo ""
        echo "✅ PyTorch upgraded successfully!"
        echo ""
        echo "You can now run training with:"
        echo "python3 train_bert.py -d clickbait_data.csv -b 24 -nw 6 --use-amp --compile-model -ga 2"
        ;;

    2)
        echo ""
        echo "Downgrading Transformers to 4.45.2..."
        echo ""
        pip install transformers==4.45.2

        echo ""
        echo "⚠️  WARNING: RTX 5080 will NOT work with PyTorch 2.2.1"
        echo "Training will run on CPU and will be very slow"
        echo ""
        echo "Run training with:"
        echo "python3 train_bert.py -d clickbait_data.csv -b 8 -nw 4"
        ;;

    3)
        echo ""
        echo "Installing PyTorch Nightly..."
        echo ""

        pip uninstall torch torchvision torchaudio -y
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

        echo ""
        echo "Verifying installation..."
        python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

        echo ""
        echo "⚠️  Nightly build may be unstable"
        ;;

    4)
        echo "Cancelled."
        exit 0
        ;;

    *)
        echo "Invalid choice."
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "Done!"
echo "=================================="
