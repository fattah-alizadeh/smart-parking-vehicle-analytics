import time

import torch


class DetectorBenchmark:

    # Input: Computation device such as CUDA or CPU.
    # Output: Initialized detector benchmark.
    # Purpose: Store the device used for timing vehicle detection models.
    def __init__(self, device):
        self.device = device


    # Input: VehicleDetector instance and one full parking image.
    # Output: Average inference time in milliseconds and frames per second.
    # Purpose: Measure repeated detection speed on the same image.
    def measure_speed(
        self,
        detector,
        image,
        warmup_runs=10,
        benchmark_runs=50
    ):
        for _ in range(warmup_runs):
            detector.detect(image)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(benchmark_runs):
            detector.detect(image)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time = end_time - start_time
        average_seconds = total_time / benchmark_runs

        latency_ms = average_seconds * 1000
        fps = 1 / average_seconds

        return latency_ms, fps