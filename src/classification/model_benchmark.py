import time
from pathlib import Path

import torch


class ModelBenchmark:

    # Input: Computation device such as CUDA or CPU.
    # Output: Initialized ModelBenchmark object.
    # Purpose: Store the device used for model performance measurements.
    def __init__(self, device):
        self.device = device


    # Input: Model and one sample image tensor.
    # Output: Average inference time in milliseconds and images processed per second.
    # Purpose: Measure the inference speed of a classification model.
    def measure_inference_speed(
        self,
        model,
        sample_image,
        warmup_runs=20,
        benchmark_runs=100
    ):
        model.eval()

        sample_image = sample_image.to(self.device)

        # Add batch dimension if a single image was provided.
        if sample_image.dim() == 3:
            sample_image = sample_image.unsqueeze(0)

        # Warm up the GPU before measuring.
        with torch.no_grad():
            for _ in range(warmup_runs):
                model(sample_image)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(benchmark_runs):
                model(sample_image)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_time = end_time - start_time

        average_seconds = total_time / benchmark_runs

        average_milliseconds = average_seconds * 1000

        images_per_second = 1 / average_seconds

        return average_milliseconds, images_per_second


    # Input: Path to a saved model file.
    # Output: Model file size in megabytes.
    # Purpose: Measure the storage size of a trained model.
    def get_model_size_mb(self, model_path):
        model_path = Path(model_path)

        size_bytes = model_path.stat().st_size

        size_mb = size_bytes / (1024 * 1024)

        return size_mb