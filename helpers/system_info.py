# This is from a previous implementation of benchmarking utilities.
import json
import platform
from typing import Any, Dict

import torch


def get_system_info() -> Dict[str, Any]:
    """Get system information for benchmarking context."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        "mps_available": (
            torch.backends.mps.is_available()
            if hasattr(torch.backends, "mps")
            else False
        ),
    }


print(json.dumps(get_system_info(), indent=2))
