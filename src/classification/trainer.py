import torch


class Trainer:

    # Input: model, optimizer, loss function, device.
    # Output: Trainer instance configured for model training.
    # Purpose: Store reusable training components for classification models.
    def __init__(self, model, optimizer, loss_function, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device


    # Input: Training DataLoader.
    # Output: Average training loss and training accuracy for one epoch.
    # Purpose: Train the model for one complete pass through the training dataset.
    def train_one_epoch(self, data_loader):
        self.model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, labels, cameras, image_paths in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.loss_function(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        average_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples

        return average_loss, accuracy


    # Input: Validation DataLoader.
    # Output: Average validation loss and validation accuracy.
    # Purpose: Evaluate the model without changing its weights.
    def evaluate(self, data_loader):
        self.model.eval()

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels, cameras, image_paths in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.loss_function(outputs, labels)

                running_loss += loss.item() * images.size(0)

                predictions = torch.argmax(outputs, dim=1)

                correct_predictions += (
                    predictions == labels
                ).sum().item()

                total_samples += labels.size(0)

        average_loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples

        return average_loss, accuracy