#!/usr/bin/env python3
"""Check what's using GPU memory."""

import subprocess
import torch

print("=" * 60)
print("GPU Memory Check")
print("=" * 60)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("CUDA not available")

print("\n" + "=" * 60)
print("Processes using GPU:")
print("=" * 60)

try:
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
except Exception as e:
    print(f"Could not run nvidia-smi: {e}")
    print("\nTry running manually: nvidia-smi")
