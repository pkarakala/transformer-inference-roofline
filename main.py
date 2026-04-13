from __future__ import annotations

from pathlib import Path

from analysis import default_sweep, summarize_total_runtime
from hardware import HardwareSpec
from plots import (
    make_intensity_plot,
    make_latency_plot,
    make_roofline_plot,
    make_runtime_breakdown_plot,
)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    hardware = HardwareSpec.default()
    detailed_df = default_sweep(hardware)
    summary_df = summarize_total_runtime(detailed_df)

    detailed_csv = output_dir / "op_metrics.csv"
    summary_csv = output_dir / "summary_metrics.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    generated = [
        make_roofline_plot(detailed_df, hardware, output_dir),
        make_latency_plot(summary_df, output_dir),
        make_intensity_plot(summary_df, output_dir),
        make_runtime_breakdown_plot(detailed_df, output_dir),
    ]

    print(f"Project: {project_dir.name}")
    print(f"Hardware: {hardware.name}")
    print(f"Detailed metrics: {detailed_csv}")
    print(f"Summary metrics: {summary_csv}")
    print("Generated plots:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
