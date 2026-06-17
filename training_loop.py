import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard

class TrainingLoop:
    """
    A class used to train a PyTorch model with a custom learning rate schedule and gradient clipping.
    """

    def __init__(self, model: nn.Module, dataloader: data.DataLoader, device: torch.device, 
                 warmup_steps: int, max_grad_norm: float = 1.0, learning_rate: float = 1e-4):
        """
        Initializes the TrainingLoop class.

        Args:
        - model (nn.Module): The PyTorch model to be trained.
        - dataloader (data.DataLoader): The data loader for the training data.
        - device (torch.device): The device to use for training (e.g., CPU or GPU).
        - warmup_steps (int): The number of warmup steps for the learning rate schedule.
        - max_grad_norm (float, optional): The maximum gradient norm for gradient clipping. Defaults to 1.0.
        - learning_rate (float, optional): The initial learning rate. Defaults to 1e-4.
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, self._lr_schedule)
        self.writer = tensorboard.SummaryWriter()

    def _lr_schedule(self, step_num: int) -> float:
        """
        Computes the learning rate at a given step number.

        Args:
        - step_num (int): The current step number.

        Returns:
        - float: The learning rate at the given step number.
        """
        return self.learning_rate * min(step_num ** -0.5, step_num * self.warmup_steps ** -1.5)

    def train_one_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch.

        Args:
        - epoch (int): The current epoch number.

        Returns:
        - float: The average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(self.dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.dataloader) + batch_idx)
        return total_loss / len(self.dataloader)

    def train(self, num_epochs: int) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
        - num_epochs (int): The number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            loss = self.train_one_epoch(epoch)
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

if __name__ == "__main__":
    # Create a dummy model, dataloader, and device
    model = nn.Linear(5, 3)
    dataloader = data.DataLoader(torch.randn(100, 5), batch_size=10)
    device = torch.device('cpu')

    # Create a TrainingLoop instance and train the model
    training_loop = TrainingLoop(model, dataloader, device, warmup_steps=100)
    training_loop.train(10)