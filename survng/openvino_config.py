"""Small shared OpenVINO policies, importable by the isolated native runner."""

from typing import Any

# Concurrent cold GPU compilation reproduced native heap corruption in the
# Intel OpenCL compiler (IGC 2.40.13). Bound compilation, not inference: request
# pools, inference streams, VA sharing, and model precision remain unchanged.
GPU_COMPILATION_NUM_THREADS = 1


def latency_compile_config(device: str) -> dict[str, Any]:
    config: dict[str, Any] = {"PERFORMANCE_HINT": "LATENCY"}
    target = device.upper()
    if target != "AUTO":
        config["NUM_STREAMS"] = "1"
    gpu = {"COMPILATION_NUM_THREADS": GPU_COMPILATION_NUM_THREADS}
    if target.split(".", 1)[0] == "GPU":
        config.update(gpu)
    elif target.split(":", 1)[0] in {"AUTO", "MULTI", "HETERO"}:
        # CPU/NPU plugins need not support GPU compilation properties.
        config["DEVICE_PROPERTIES"] = {"GPU": gpu}
    return config
