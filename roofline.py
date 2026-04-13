from __future__ import annotations

import numpy as np
import pandas as pd

from hardware import HardwareSpec


def roofline_curve(hardware: HardwareSpec, min_intensity: float = 1e-2, max_intensity: float = 1e4) -> pd.DataFrame:
    intensities = np.logspace(np.log10(min_intensity), np.log10(max_intensity), 300)
    attainable = np.minimum(hardware.peak_flops, intensities * hardware.memory_bandwidth)
    return pd.DataFrame(
        {
            "arithmetic_intensity": intensities,
            "attainable_flops_per_s": attainable,
        }
    )


def achieved_performance(df: pd.DataFrame) -> pd.DataFrame:
    perf = df.copy()
    perf["achieved_flops_per_s"] = perf["flops"] / perf["runtime_s"].clip(lower=1e-18)
    return perf
