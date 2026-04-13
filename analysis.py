from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Sequence

import pandas as pd

from hardware import HardwareSpec
from ops import OP_REGISTRY, WorkloadConfig


def analyze_workload(config: WorkloadConfig, hardware: HardwareSpec) -> pd.DataFrame:
    rows = []
    for op_name, op_fn in OP_REGISTRY.items():
        metrics = op_fn(config, hardware)
        row = {
            "op_name": op_name,
            "phase": config.phase,
            "batch_size": config.batch_size,
            "sequence_length": config.sequence_length,
            "hidden_dim": config.hidden_dim,
            "active_tokens": config.active_tokens,
            "kv_length": config.kv_length,
            "hardware": hardware.name,
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_workloads(
    hardware: HardwareSpec,
    sequence_lengths: Sequence[int],
    batch_sizes: Sequence[int],
    hidden_dims: Sequence[int],
    phases: Iterable[str] = ("prefill", "decode"),
) -> pd.DataFrame:
    frames = []
    for phase in phases:
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                for hidden_dim in hidden_dims:
                    config = WorkloadConfig(
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        hidden_dim=hidden_dim,
                        phase=phase,
                    )
                    frames.append(analyze_workload(config, hardware))
    return pd.concat(frames, ignore_index=True)


def summarize_total_runtime(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["phase", "batch_size", "sequence_length", "hidden_dim", "hardware"]
    summary = (
        df.groupby(group_cols, as_index=False)
        .agg(
            total_runtime_s=("runtime_s", "sum"),
            total_flops=("flops", "sum"),
            total_bytes_moved=("bytes_moved", "sum"),
        )
        .sort_values(group_cols)
    )
    summary["overall_arithmetic_intensity"] = (
        summary["total_flops"] / summary["total_bytes_moved"].clip(lower=1.0)
    )
    return summary


def default_sweep(hardware: HardwareSpec) -> pd.DataFrame:
    return sweep_workloads(
        hardware=hardware,
        sequence_lengths=[1, 8, 16, 32, 64, 128, 256, 512, 1024],
        batch_sizes=[1, 4, 8],
        hidden_dims=[512, 1024, 2048],
        phases=("prefill", "decode"),
    )
