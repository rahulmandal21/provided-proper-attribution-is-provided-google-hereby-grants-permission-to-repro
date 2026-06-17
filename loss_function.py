import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossFunction(nn.Module):
    """
    A PyTorch module that implements a cross-entropy loss function for sequence transduction tasks.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initializes the CrossEntropyLossFunction module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The cross-entropy loss.
        """
        return self.criterion(predictions, targets)


class SequenceTransductionLoss(nn.Module):
    """
    A PyTorch module that implements a loss function for sequence transduction tasks.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initializes the SequenceTransductionLoss module.

        Args:
        - num_classes (int): The number of classes in the classification problem.
        - smoothing (float, optional): The label smoothing factor. Defaults to 0.1.
        """
        super().__init__()
        self.loss_function = CrossEntropyLossFunction(num_classes, smoothing)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between the predictions and targets.

        Args:
        - predictions (torch.Tensor): The predicted logits.
        - targets (torch.Tensor): The true labels.

        Returns:
        - torch.Tensor: The loss.
        """
        return self.loss_function(predictions, targets)


if __name__ == "__main__":
    # Create a dummy dataset
    predictions = torch.randn(10, 5, 3)
    targets = torch.randint(0, 3, (10, 5))

    # Initialize the loss function
    loss_function = SequenceTransductionLoss(num_classes=3)

    # Compute the loss
    loss = loss_function(predictions, targets)

    print("Loss:", loss.item())