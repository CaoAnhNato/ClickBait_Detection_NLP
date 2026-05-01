"""
Script to check the structure of the model weights file
"""
import torch

def check_checkpoint_structure(weight_path):
    """Check and print the structure of checkpoint"""
    print(f"Loading checkpoint from: {weight_path}")

    try:
        checkpoint = torch.load(weight_path, map_location='cpu')

        print("\n" + "="*60)
        print("Checkpoint Structure")
        print("="*60 + "\n")

        if isinstance(checkpoint, dict):
            print(f"Type: Dictionary with {len(checkpoint)} keys\n")
            print("Keys in checkpoint:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: Tensor with shape {value.shape}")
                elif isinstance(value, dict):
                    print(f"  - {key}: Dictionary with {len(value)} items")
                    # Show some sub-keys if it's a state dict
                    sub_keys = list(value.keys())[:5]
                    for sub_key in sub_keys:
                        if isinstance(value[sub_key], torch.Tensor):
                            print(f"      - {sub_key}: Tensor with shape {value[sub_key].shape}")
                    if len(value) > 5:
                        print(f"      ... and {len(value) - 5} more items")
                else:
                    print(f"  - {key}: {type(value)}")

        elif isinstance(checkpoint, torch.nn.Module):
            print("Type: PyTorch Module (model object)")
            print("\nModel structure:")
            print(checkpoint)

        else:
            print(f"Type: {type(checkpoint)}")
            print("\nContent preview:")
            print(checkpoint)

        print("\n" + "="*60)

    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    weight_path = "weight/best_teachermodel.pth"
    check_checkpoint_structure(weight_path)
