import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.model_benchmark import ModelBenchmark
from src.classification.model_factory import ModelFactory
from src.classification.parking_dataset import ParkingSpaceDataset
from src.classification.transforms import build_eval_transform


# Input: Model, saved model path, sample image, benchmark object, and device.
# Output: Dictionary containing model name, size, latency, and throughput.
# Purpose: Load and benchmark one trained parking classification model.
def benchmark_model(
    model_name,
    model,
    model_path,
    sample_image,
    benchmark,
    device
):
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)

    latency_ms, images_per_second = (
        benchmark.measure_inference_speed(
            model,
            sample_image
        )
    )

    model_size_mb = benchmark.get_model_size_mb(
        model_path
    )

    return {
        "name": model_name,
        "size_mb": model_size_mb,
        "latency_ms": latency_ms,
        "images_per_second": images_per_second
    }


# Input: None.
# Output: Prints benchmark results for MobileNetV3-Small and ResNet18.
# Purpose: Compare trained parking classifiers on the same GPU and image.
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    eval_transform = build_eval_transform()

    # We use one Camera A image for both models.
    dataset = ParkingSpaceDataset(
        dataset_root=PROJECT_ROOT / "data/public/cnrpark",
        cameras=["A"],
        class_to_index={
            "Free": 0,
            "Busy": 1
        },
        transform=eval_transform
    )

    sample_image, _, _, _ = dataset[0]

    model_folder = (
        PROJECT_ROOT
        / "results"
        / "models"
    )

    mobilenet_path = (
        model_folder
        / "mobilenet_v3_cameraB_to_cameraA_augmented_best.pth"
    )

    resnet_path = (
        model_folder
        / "resnet18_cameraB_to_cameraA_augmented_best.pth"
    )

    benchmark = ModelBenchmark(
        device=device
    )

    mobilenet_result = benchmark_model(
        model_name="MobileNetV3-Small",
        model=ModelFactory.create_mobilenet_v3_small(
            num_classes=2
        ),
        model_path=mobilenet_path,
        sample_image=sample_image,
        benchmark=benchmark,
        device=device
    )

    resnet_result = benchmark_model(
        model_name="ResNet18",
        model=ModelFactory.create_resnet18(
            num_classes=2
        ),
        model_path=resnet_path,
        sample_image=sample_image,
        benchmark=benchmark,
        device=device
    )

    print("\nMODEL BENCHMARK")
    print("---------------")

    for result in [
        mobilenet_result,
        resnet_result
    ]:
        print("\nModel:", result["name"])
        print(
            f"Model size: {result['size_mb']:.2f} MB"
        )
        print(
            f"Average latency: {result['latency_ms']:.3f} ms/image"
        )
        print(
            f"Throughput: {result['images_per_second']:.2f} images/sec"
        )


if __name__ == "__main__":
    main()