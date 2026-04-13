from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSpec:
    """Simple roofline hardware model."""

    name: str
    peak_flops: float
    memory_bandwidth: float

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity where the roofline changes from memory- to compute-bound."""
        return self.peak_flops / self.memory_bandwidth

    @classmethod
    def default(cls) -> "HardwareSpec":
        # Interview-friendly toy accelerator numbers.
        # 120 TFLOP/s and 1.5 TB/s are intentionally rounded.
        return cls(
            name="Toy Accelerator",
            peak_flops=120e12,
            memory_bandwidth=1.5e12,
        )
