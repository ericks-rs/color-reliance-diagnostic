import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok:", (x @ x).sum().item() is not None)
else:
    print("GPU GATE FAILED: CUDA not available. Install cu128 build before continuing.")
