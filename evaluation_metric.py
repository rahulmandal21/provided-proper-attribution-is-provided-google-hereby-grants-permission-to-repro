import torch
import torch.nn as nn
import sacrebleu
from torch.utils.tensorboard import SummaryWriter
import csv
from typing import List

class BLEUEvaluator(nn.Module):
    """
    A PyTorch module for evaluating the performance of a machine translation model using the BLEU score.
    """

    def __init__(self):
        """
        Initializes the BLEUEvaluator module.
        """
        super().__init__()
        self.writer = SummaryWriter()

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """
        Computes the BLEU score for the given predictions and references.

        Args:
        predictions (List[str]): A list of predicted translations.
        references (List[List[str]]): A list of reference translations.

        Returns:
        float: The BLEU score.
        """
        bleu = sacrebleu.corpus_bleu(predictions, references)
        return bleu.score

    def log_metrics(self, epoch: int, bleu_score: float):
        """
        Logs the BLEU score to TensorBoard.

        Args:
        epoch (int): The current epoch.
        bleu_score (float): The BLEU score.
        """
        self.writer.add_scalar('BLEU Score', bleu_score, epoch)

    def save_to_csv(self, epoch: int, bleu_score: float, filename: str):
        """
        Saves the BLEU score to a CSV file.

        Args:
        epoch (int): The current epoch.
        bleu_score (float): The BLEU score.
        filename (str): The filename to save to.
        """
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['epoch', 'bleu_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({'epoch': epoch, 'bleu_score': bleu_score})

    def evaluate(self, predictions: List[str], references: List[List[str]], epoch: int, filename: str):
        """
        Evaluates the model's performance using the BLEU score and logs the result.

        Args:
        predictions (List[str]): A list of predicted translations.
        references (List[List[str]]): A list of reference translations.
        epoch (int): The current epoch.
        filename (str): The filename to save the result to.
        """
        bleu_score = self.compute_bleu(predictions, references)
        self.log_metrics(epoch, bleu_score)
        self.save_to_csv(epoch, bleu_score, filename)
        return bleu_score


if __name__ == "__main__":
    evaluator = BLEUEvaluator()
    predictions = ["This is a test sentence", "Another test sentence"]
    references = [["This is a test sentence", "This is another test sentence"], ["Another test sentence", "Yet another test sentence"]]
    epoch = 1
    filename = 'bleu_scores.csv'
    bleu_score = evaluator.evaluate(predictions, references, epoch, filename)
    print(f"BLEU Score: {bleu_score}")