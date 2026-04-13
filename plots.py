from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from hardware import HardwareSpec
from roofline import achieved_performance, roofline_curve


PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.style.use("seaborn-v0_8-whitegrid")


def _phase_color(phase: str) -> str:
    return {"prefill": "#1f77b4", "decode": "#d62728"}.get(phase, "#333333")


def make_roofline_plot(df: pd.DataFrame, hardware: HardwareSpec, output_dir: Path) -> Path:
    roof = roofline_curve(hardware)
    perf = achieved_performance(df)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(
        roof["arithmetic_intensity"],
        roof["attainable_flops_per_s"],
        color="#111111",
        linewidth=2.5,
        label=f"{hardware.name} roofline",
    )

    for phase in sorted(perf["phase"].unique()):
        phase_df = perf[perf["phase"] == phase]
        ax.scatter(
            phase_df["arithmetic_intensity"],
            phase_df["achieved_flops_per_s"],
            s=50,
            alpha=0.7,
            label=f"{phase} ops",
            color=_phase_color(phase),
        )

    ax.axvline(hardware.ridge_point, color="#666666", linestyle="--", linewidth=1.5, label="ridge point")
    ax.set_title("Transformer Inference Roofline")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Performance (FLOP/s)")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "roofline_plot.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_latency_plot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    plot_df = summary_df[
        (summary_df["batch_size"] == summary_df["batch_size"].min())
        & (summary_df["hidden_dim"] == summary_df["hidden_dim"].median())
    ].copy()

    for phase in sorted(plot_df["phase"].unique()):
        phase_df = plot_df[plot_df["phase"] == phase].sort_values("sequence_length")
        ax.plot(
            phase_df["sequence_length"],
            phase_df["total_runtime_s"] * 1e3,
            marker="o",
            linewidth=2,
            label=f"{phase}",
            color=_phase_color(phase),
        )

    ax.set_title("Latency vs Sequence Length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Estimated latency (ms)")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "latency_vs_sequence_length.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_intensity_plot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))

    plot_df = summary_df[
        (summary_df["batch_size"] == summary_df["batch_size"].min())
        & (summary_df["hidden_dim"] == summary_df["hidden_dim"].median())
    ].copy()

    for phase in sorted(plot_df["phase"].unique()):
        phase_df = plot_df[plot_df["phase"] == phase].sort_values("sequence_length")
        ax.plot(
            phase_df["sequence_length"],
            phase_df["overall_arithmetic_intensity"],
            marker="o",
            linewidth=2,
            label=f"{phase}",
            color=_phase_color(phase),
        )

    ax.set_title("Arithmetic Intensity vs Sequence Length")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Overall arithmetic intensity (FLOPs / byte)")
    ax.legend()
    fig.tight_layout()

    output_path = output_dir / "arithmetic_intensity_vs_sequence_length.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_runtime_breakdown_plot(df: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = df[
        (df["batch_size"] == df["batch_size"].min())
        & (df["hidden_dim"] == df["hidden_dim"].median())
    ].copy()

    pivot_df = (
        plot_df.pivot_table(
            index=["phase", "sequence_length"],
            columns="op_name",
            values="runtime_s",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, phase in zip(axes, ["prefill", "decode"]):
        phase_df = pivot_df.loc[phase]
        phase_df.mul(1e3).plot(
            kind="bar",
            stacked=True,
            ax=ax,
            colormap="tab20c",
            width=0.8,
        )
        ax.set_title(f"{phase.capitalize()} Runtime Breakdown")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Estimated runtime (ms)")
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    output_path = output_dir / "stacked_runtime_breakdown.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
