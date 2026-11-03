import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    resnet18,
    ResNet18_Weights
)


class ModelFactory:

    # Input: Number of output classes.
    # Output: MobileNetV3-Small model configured for parking classification.
    # Purpose: Create a pretrained MobileNetV3-Small classifier for Free/Busy prediction.
    @staticmethod
    def create_mobilenet_v3_small(num_classes: int = 2):
        weights = MobileNet_V3_Small_Weights.DEFAULT

        model = mobilenet_v3_small(
            weights=weights
        )

        input_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            input_features,
            num_classes
        )

        return model


    # Input: Number of output classes.
    # Output: ResNet18 model configured for parking classification.
    # Purpose: Create a pretrained ResNet18 classifier for Free/Busy prediction.
    @staticmethod
    def create_resnet18(num_classes: int = 2):
        weights = ResNet18_Weights.DEFAULT

        model = resnet18(
            weights=weights
        )

        input_features = model.fc.in_features

        model.fc = nn.Linear(
            input_features,
            num_classes
        )

        return model