"""Validate installed dependencies, local imports, and real CUDA execution."""

import importlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
result = {"python": platform.python_version(), "executable": sys.executable}
failures = []


def check(name, operation):
    try:
        result[name] = operation()
        print(f"PASS {name}: {result[name]}", flush=True)
    except Exception as exc:
        result[name] = {"error": f"{type(exc).__name__}: {exc}"}
        failures.append(name)
        print(f"FAIL {name}: {result[name]['error']}", flush=True)


def package_check():
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"], text=True, capture_output=True
    )
    if proc.returncode:
        raise RuntimeError(proc.stdout + proc.stderr)
    return proc.stdout.strip()


check("pip_check", package_check)
for module in (
    "torch", "torchvision", "torchaudio", "numpy", "scipy", "pandas",
    "sklearn", "mne", "h5py", "transformers", "peft", "timm", "pyhealth",
    "pytorch_lightning", "torcheeg", "braindecode", "mamba_ssm",
    "src.data_loader", "src.utils", "src.model", "src.train", "src.biot",
):
    check(f"import:{module}", lambda module=module: (
        getattr(importlib.import_module(module), "__version__", "OK")
    ))


def gpu_check():
    import torch

    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", requires_grad=True)
    loss = (a @ a.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    return {
        "name": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "architectures": torch.cuda.get_arch_list(),
        "forward_backward": "OK",
    }


def cbramod_check():
    import torch
    from src.cbramod import CBraMod

    model = CBraMod(n_layer=1, seq_len=4).cuda()
    inputs = torch.randn(2, 4, 4, 200, device="cuda")
    outputs = model(inputs)
    assert outputs.shape == inputs.shape, outputs.shape
    outputs.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(outputs).all()
    return {"output_shape": list(outputs.shape), "forward_backward": "OK"}


def mamba_check():
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    batch, channels, length, states = 2, 16, 64, 8
    u = torch.randn(batch, channels, length, device="cuda", requires_grad=True)
    delta = torch.randn_like(u).requires_grad_()
    a = -torch.rand(channels, states, device="cuda")
    b = torch.randn(batch, states, length, device="cuda")
    c = torch.randn_like(b)
    output = selective_scan_fn(u, delta, a, b, c, delta_softplus=True)
    output.square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all() and torch.isfinite(u.grad).all()
    return {"output_shape": list(output.shape), "forward_backward": "OK"}


def decord_check():
    import cv2
    import decord
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="eegpt-video-check-") as temporary:
        video = str(Path(temporary) / "test.avi")
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
        assert writer.isOpened(), "OpenCV test video writer unavailable"
        for value in (0, 100, 200):
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
        writer.release()
        reader = decord.VideoReader(video, ctx=decord.cpu(0))
        frames = reader.get_batch([0, 1, 2]).asnumpy()
        assert frames.shape == (3, 48, 64, 3), frames.shape
        return {"version": decord.__version__, "decoded_shape": list(frames.shape)}


check("cmake", lambda: subprocess.check_output(
    [str(Path(sys.executable).with_name("cmake")), "--version"], text=True
).splitlines()[0])
check("decord_video_decode", decord_check)
check("cuda", gpu_check)
check("cbramod_cuda", cbramod_check)
check("mamba_selective_scan_cuda", mamba_check)
result["failed_checks"] = failures
Path(__file__).with_name("validation.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
)
sys.exit(bool(failures))
