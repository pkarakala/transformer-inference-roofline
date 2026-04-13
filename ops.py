from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from hardware import HardwareSpec


@dataclass(frozen=True)
class WorkloadConfig:
    batch_size: int
    sequence_length: int
    hidden_dim: int
    phase: str
    bytes_per_element: int = 2

    @property
    def kv_length(self) -> int:
        """Decode attends over prior tokens; prefill attends over the prompt length."""
        if self.phase == "decode":
            return max(self.sequence_length - 1, 1)
        return self.sequence_length

    @property
    def active_tokens(self) -> int:
        """Prefill processes the full prompt; decode handles one token step."""
        return self.sequence_length if self.phase == "prefill" else 1


def _runtime_fields(flops: float, bytes_moved: float, hardware: HardwareSpec) -> Dict[str, float | str]:
    arithmetic_intensity = flops / max(bytes_moved, 1.0)
    compute_time = flops / hardware.peak_flops
    memory_time = bytes_moved / hardware.memory_bandwidth
    runtime = max(compute_time, memory_time)
    bottleneck = "compute-bound" if compute_time >= memory_time else "memory-bound"
    return {
        "flops": flops,
        "bytes_moved": bytes_moved,
        "arithmetic_intensity": arithmetic_intensity,
        "compute_time_s": compute_time,
        "memory_time_s": memory_time,
        "runtime_s": runtime,
        "bottleneck": bottleneck,
    }


def linear_projection(config: WorkloadConfig, hardware: HardwareSpec) -> Dict[str, float | str]:
    b, t, h, dtype = (
        config.batch_size,
        config.active_tokens,
        config.hidden_dim,
        config.bytes_per_element,
    )
    # Matrix multiply approximation: [B*T, H] x [H, H]
    flops = 2.0 * b * t * h * h
    # Read activations and weights, write outputs.
    bytes_moved = dtype * (b * t * h + h * h + b * t * h)
    return _runtime_fields(flops, bytes_moved, hardware)


def attention_scores(config: WorkloadConfig, hardware: HardwareSpec) -> Dict[str, float | str]:
    b, t, h, kv, dtype = (
        config.batch_size,
        config.active_tokens,
        config.hidden_dim,
        config.kv_length,
        config.bytes_per_element,
    )
    # QK^T approximation per batch: [T, H] x [H, KV]
    flops = 2.0 * b * t * kv * h
    # Read Q and K, write score matrix.
    bytes_moved = dtype * (b * t * h + b * kv * h + b * t * kv)
    return _runtime_fields(flops, bytes_moved, hardware)


def attention_value_aggregation(config: WorkloadConfig, hardware: HardwareSpec) -> Dict[str, float | str]:
    b, t, h, kv, dtype = (
        config.batch_size,
        config.active_tokens,
        config.hidden_dim,
        config.kv_length,
        config.bytes_per_element,
    )
    # Attention output approximation: [T, KV] x [KV, H]
    flops = 2.0 * b * t * kv * h
    # Read attention weights and V, write context output.
    bytes_moved = dtype * (b * t * kv + b * kv * h + b * t * h)
    return _runtime_fields(flops, bytes_moved, hardware)


def mlp_output_projection(config: WorkloadConfig, hardware: HardwareSpec) -> Dict[str, float | str]:
    b, t, h, dtype = (
        config.batch_size,
        config.active_tokens,
        config.hidden_dim,
        config.bytes_per_element,
    )
    intermediate = 4 * h
    # Two GEMMs: H -> 4H and 4H -> H. This ignores activation function cost for simplicity.
    flops = 2.0 * b * t * h * intermediate + 2.0 * b * t * intermediate * h
    bytes_moved = dtype * (
        b * t * h + h * intermediate + b * t * intermediate + intermediate * h + b * t * h
    )
    return _runtime_fields(flops, bytes_moved, hardware)


def kv_cache_update(config: WorkloadConfig, hardware: HardwareSpec) -> Dict[str, float | str]:
    b, t, h, dtype = (
        config.batch_size,
        config.active_tokens,
        config.hidden_dim,
        config.bytes_per_element,
    )
    # Treat KV-cache update as mostly a memory traffic event with negligible math.
    flops = float(b * t * h)
    # Write K and V for the active token(s); include a small read term to model staging.
    bytes_moved = dtype * (b * t * h + 2 * b * t * h)
    return _runtime_fields(flops, bytes_moved, hardware)


OP_REGISTRY = {
    "linear_projection": linear_projection,
    "attention_scores": attention_scores,
    "attention_value_aggregation": attention_value_aggregation,
    "mlp_output_projection": mlp_output_projection,
    "kv_cache_update": kv_cache_update,
}
