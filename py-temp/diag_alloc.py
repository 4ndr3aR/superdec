"""Confirm which torch env var controls expandable_segments in this torch (2.9.1), and whether it
is currently taking effect. get_allocator_backend() == 'cudaMallocAsync' when expandable_segments
is active. Run on pod py-3.13."""
os_ = None
import os, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for var in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
    print(f"  env[{var}] = {os.environ.get(var)!r}")
torch.cuda.init()
print("allocator backend =", torch.cuda.get_allocator_backend())
print("is expandable =", torch.cuda.get_allocator_backend() == "cudaMallocAsync")
