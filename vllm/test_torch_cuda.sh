# Sanity check that torch really works on the GPU inside the image (needs --runtime=nvidia).
# Usage: bash test_torch_cuda.sh [image]   (default: vllm-jetson)
docker run -i --rm --runtime=nvidia --entrypoint python3 "${1:-vllm-jetson}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
c = (a.cuda() @ b.cuda()); torch.cuda.synchronize()
assert torch.allclose(c.cpu(), a @ b, atol=1e-2), "fp32 matmul mismatch"
h = (a.cuda().half() @ b.cuda().half()); torch.cuda.synchronize()
assert torch.allclose(h.float().cpu(), a @ b, atol=1.0), "fp16 matmul mismatch"
print("matmul fp32/fp16 ok")
try:
    import torchvision
    from torchvision.ops import nms
    nms(torch.rand(8, 4).cuda(), torch.rand(8).cuda(), 0.5); torch.cuda.synchronize()
    print("torchvision", torchvision.__version__, "nms ok")
except ImportError:
    print("torchvision not installed (skipped)")
print("mem", [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()], "GiB free/total")
print("ALL OK")
PY
