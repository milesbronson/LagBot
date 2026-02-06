#!/usr/bin/env python3
"""
Quick GPU check for PyTorch training on Apple Silicon
"""

import sys

def check_gpu():
    print("=" * 70)
    print("GPU Availability Check for LagBot Training")
    print("=" * 70)
    print()

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed!")
        print("  Run: pip3 install -r requirements.txt")
        sys.exit(1)

    print()
    print("Available Devices:")
    print("-" * 70)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        recommended = "cuda"
    else:
        print("✗ CUDA not available")

    # Check MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        print(f"✓ Metal Performance Shaders (MPS) available")
        print(f"  Apple Silicon GPU acceleration enabled")
        recommended = "mps"
    else:
        print("✗ MPS not available")

    # CPU is always available
    print(f"✓ CPU available")

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        recommended = "cpu"
        print()
        print("⚠️  WARNING: No GPU acceleration detected!")
        print("   Training will be slower on CPU.")

    print()
    print("=" * 70)
    print(f"Recommended device: {recommended}")
    print("=" * 70)
    print()

    # Test tensor creation
    print("Testing tensor creation on recommended device...")
    try:
        if recommended == "mps":
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✓ Successfully created and multiplied tensors on MPS")
        elif recommended == "cuda":
            device = torch.device("cuda")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✓ Successfully created and multiplied tensors on CUDA")
        else:
            device = torch.device("cpu")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✓ Successfully created and multiplied tensors on CPU")

        print()
        print(f"🚀 Ready to train with {recommended.upper()} acceleration!")

    except Exception as e:
        print(f"✗ Error testing device: {e}")
        print(f"   Falling back to CPU")

    print()
    print("Next steps:")
    print("  1. Run: python3 train.py")
    print("  2. Monitor: tensorboard --logdir ./logs/")
    print()

if __name__ == "__main__":
    check_gpu()
