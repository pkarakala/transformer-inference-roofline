# transformer-inference-roofline

`transformer-inference-roofline` is a toy Python project for reasoning about transformer inference performance using simple roofline-style analysis. The goal is not cycle accuracy; the goal is to give a framework for discussing which inference operations are likely compute-bound or memory-bound as model and workload parameters change.

## Project Goal

This project models a few first-order transformer inference operations:

- linear projection
- attention score computation
- attention-weighted value aggregation
- MLP/output projection
- KV-cache update

For each operation, it estimates:

- FLOPs
- bytes moved
- arithmetic intensity
- compute-limited runtime
- memory-limited runtime
- overall runtime estimate
- bottleneck classification

It also sweeps over sequence length, batch size, and hidden dimension, and compares prefill versus decode behavior.

## Example Outputs

### Roofline Plot

![Roofline plot](assets/roofline_plot.png)

### Latency vs Sequence Length

![Latency vs sequence length](assets/latency_vs_sequence_length.png)

### Arithmetic Intensity vs Sequence Length

![Arithmetic intensity vs sequence length](assets/arithmetic_intensity_vs_sequence_length.png)

### Stacked Op Runtime Breakdown

![Stacked runtime breakdown](assets/stacked_runtime_breakdown.png)

## Files

- `hardware.py`: toy hardware roofline model
- `ops.py`: simplified transformer op formulas
- `analysis.py`: per-op analysis and workload sweeps
- `roofline.py`: roofline and achieved-performance helpers
- `plots.py`: plotting utilities
- `main.py`: entrypoint that runs the sweep and saves outputs
- `assets/`: tracked images displayed in this README

## Analytical Assumptions

The implementation intentionally uses simple approximations that are easy to explain in an interview:

1. Dense GEMMs are modeled with the standard `2 * M * N * K` FLOP estimate.
2. Memory traffic is modeled as reads of inputs/weights plus writes of outputs.
3. Elements are assumed to be 2 bytes each, roughly corresponding to FP16/BF16 inference.
4. KV-cache update is treated as mostly a memory movement event with negligible math.
5. Prefill processes the full sequence, while decode processes one new token that attends over the existing KV cache.
6. Overall runtime for an op is estimated with:

   `runtime = max(flops / peak_flops, bytes / memory_bandwidth)`

This is a classic first-order roofline approximation.

## Limitations

This is deliberately not a production performance model. It ignores many real effects, including:

- kernel launch overheads
- fusion
- softmax cost
- tensor parallel communication
- cache hierarchy effects
- thread/block scheduling effects
- head count and per-head dimensions
- register/shared-memory pressure
- sparsity and quantization details

The point is clarity, not realism.

## Why This Is Relevant

This kind of analysis is useful for inference performance discussions because it helps answer questions like:

- Which ops are fundamentally limited by memory bandwidth?
- Which ops benefit most from higher arithmetic intensity?
- Why does decode often look more memory-sensitive than prefill?
- How do longer contexts change the balance between attention and MLP work?

That makes it a good conversation piece for systems, performance, and hardware-software co-design interviews.

## Reading The Results

- The roofline plot shows each modeled op against the hardware ceiling.
- The latency plot shows how estimated end-to-end latency changes with context length.
- The arithmetic intensity plot shows whether longer sequences push the workload toward compute-bound or memory-bound regimes.
- The stacked runtime plot shows which operations dominate total estimated runtime in prefill versus decode.

