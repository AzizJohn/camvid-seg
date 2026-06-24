import torch
import time

# 1. Define how much VRAM you want to test (in gigabytes)
GB_TO_ALLOCATE = 6

# 2. Convert GB to bytes (1 GB = 1024^3 bytes)
bytes_to_allocate = int(GB_TO_ALLOCATE * (1024 ** 3))
elements_to_allocate = bytes_to_allocate // 4  # Float32 takes 4 bytes

print(f"Attempting to allocate {GB_TO_ALLOCATE} GB of GPU memory...")

try:
    # 3. Allocate tensor on the GPU
    dummy_tensor = torch.empty(elements_to_allocate, dtype=torch.float32, device='cuda')
    print(f"Memory successfully allocated. xohlagancha ...")
    print(f"Press Ctrl+C to stop the process and free memory.")

    # 4. Infinite loop to keep the process alive
    while True:
        time.sleep(1)

except RuntimeError as e:
    print(f"Allocation failed: {e}. You may have asked for more memory than your GPU has available.")
