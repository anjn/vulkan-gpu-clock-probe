# clock-probe

A minimal Vulkan compute program that dispatches an ALU-bound kernel, uses GPU timestamp queries to measure per-dispatch duration, and logs CSV. Variations in runtime correlate strongly with GPU core clock changes under ALU-bound conditions.

## Build
- Requires Vulkan SDK (glslc, headers, loader)

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run
```

clock-probe.exe --iters 20000 --invocations 1048576 --loops 600 --interval_ms 1000 --csv out.csv

```
Tune `--iters` so a single dispatch lasts ~30–80 ms. Too short => noisy; too long => slow sampling.

## CSV columns
- `wall_time_iso`: Host timestamp (local time)
- `duration_ms`: GPU dispatch time from Vulkan timestamps
- `rel_freq`: Baseline_duration / current_duration (proxy for relative clock)
- `iters`, `invocations`: Workload parameters for reproducibility

## Notes
- The kernel writes one uint per thread to avoid dead-code elimination while staying ALU-bound.
- We don’t read back that buffer during sampling to avoid PCIe noise.
- For even cleaner results: run on a quiet system, disable background apps, use a dedicated compute queue.

## Extending (optional)
- Add ADLX polling (SCLK/MCLK/temperature) to log telemetry alongside the durations and compute correlations.
- Use `VK_EXT_calibrated_timestamps` if you need tighter host↔device time alignment.
