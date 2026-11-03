from torchvision import transforms


# Input: None.
# Output: Training transform pipeline with image augmentation.
# Purpose: Make training images more varied so the model generalizes better to new cameras.
def build_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(
                degrees=8
            ),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),

            transforms.ToTensor(),

            # ImageNet RGB mean and standard deviation values used to normalize images
            # because MobileNetV3 uses pretrained ImageNet weights.
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# Input: None.
# Output: Evaluation transform pipeline without random augmentation.
# Purpose: Prepare validation and test images consistently for fair evaluation.
def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

            # ImageNet RGB mean and standard deviation values used to normalize images
            # because MobileNetV3 uses pretrained ImageNet weights.
            # the following numbers are weights pretrained on the ImageNet dataset
            # The three mean values correspond to: Red mean   = 0.485 , Green mean = 0.456, and  Blue mean  = 0.406
            # Red std   = 0.229, Green std = 0.224, Blue std  = 0.225
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
    )