from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import tensorflow as tf


@dataclass(frozen=True)
class RuntimeConfig:
    device_preference: str = "auto"
    enable_mixed_precision: bool = False
    enable_xla: bool = False


def configure_tensorflow_runtime(config: RuntimeConfig) -> dict[str, object]:
    if config.enable_xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # Device initialization may already be locked by the time this runs.
            pass

    selected = "CPU"
    if config.device_preference == "gpu":
        if not gpus:
            raise RuntimeError("GPU was requested, but TensorFlow did not detect any GPU devices.")
        selected = "GPU"
    elif config.device_preference == "auto" and gpus:
        selected = "GPU"

    if config.enable_mixed_precision and selected == "GPU":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        mixed_precision_policy = "mixed_float16"
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        mixed_precision_policy = "float32"

    return {
        "selected_device": selected,
        "visible_gpus": [device.name for device in gpus],
        "mixed_precision_policy": mixed_precision_policy,
        "xla_enabled": bool(config.enable_xla),
        "device_preference": config.device_preference,
        "platform": os.name,
    }


def runtime_summary_text(config: RuntimeConfig, status: dict[str, object]) -> str:
    details = asdict(config) | status
    return (
        "TensorFlow runtime | "
        f"preferred={details['device_preference']} | "
        f"selected={details['selected_device']} | "
        f"gpus={len(details['visible_gpus'])} | "
        f"mixed_precision={details['mixed_precision_policy']} | "
        f"xla={details['xla_enabled']}"
    )
