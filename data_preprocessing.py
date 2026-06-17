import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class DataPreprocessing:
    """
    A class used to preprocess the input and output sequences.
    
    Attributes:
    ----------
    d_model : int
        The dimension of the embedding vectors.
    vocab_size : int
        The size of the vocabulary.
    max_length : int
        The maximum length of the sequences.
    """

    def __init__(self, d_model: int, vocab_size: int, max_length: int):
        """
        Initializes the DataPreprocessing class.
        
        Parameters:
        ----------
        d_model : int
            The dimension of the embedding vectors.
        vocab_size : int
            The size of the vocabulary.
        max_length : int
            The maximum length of the sequences.
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, d_model)

    def preprocess(self, sequences: list) -> torch.Tensor:
        """
        Preprocesses the input sequences by converting them to tensors and padding them to the same length.
        
        Parameters:
        ----------
        sequences : list
            A list of sequences to be preprocessed.
        
        Returns:
        -------
        torch.Tensor
            A tensor containing the preprocessed sequences.
        """
        tensor_sequences = [torch.tensor(seq) for seq in sequences]
        padded_sequences = pad_sequence(tensor_sequences, batch_first=True, padding_value=0)
        return padded_sequences

    def embed(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input sequences using the learned embeddings.
        
        Parameters:
        ----------
        sequences : torch.Tensor
            A tensor containing the input sequences.
        
        Returns:
        -------
        torch.Tensor
            A tensor containing the embedded sequences.
        """
        embedded_sequences = self.embedding(sequences)
        return embedded_sequences


class TokenizedTextDataset(Dataset):
    """
    A custom dataset class to load and preprocess the data.
    
    Attributes:
    ----------
    sequences : list
        A list of sequences to be loaded and preprocessed.
    """

    def __init__(self, sequences: list):
        """
        Initializes the TokenizedTextDataset class.
        
        Parameters:
        ----------
        sequences : list
            A list of sequences to be loaded and preprocessed.
        """
        self.sequences = [torch.tensor(seq) for seq in sequences]

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        
        Returns:
        -------
        int
            The length of the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a sequence from the dataset.
        
        Parameters:
        ----------
        idx : int
            The index of the sequence to be returned.
        
        Returns:
        -------
        torch.Tensor
            The sequence at the specified index.
        """
        return self.sequences[idx]


def collate_fn(batch: list) -> torch.Tensor:
    """
    A custom collate function to pad the sequences in a batch to the same length.
    
    Parameters:
    ----------
    batch : list
        A list of sequences to be padded.
    
    Returns:
    -------
    torch.Tensor
        A tensor containing the padded sequences.
    """
    return pad_sequence(batch, batch_first=True, padding_value=0)


if __name__ == "__main__":
    # Create a DataPreprocessing instance
    preprocessing = DataPreprocessing(d_model=128, vocab_size=1000, max_length=50)
    
    # Create some sample sequences
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    
    # Preprocess the sequences
    padded_sequences = preprocessing.preprocess(sequences)
    
    # Embed the preprocessed sequences
    embedded_sequences = preprocessing.embed(padded_sequences)
    
    # Create a TokenizedTextDataset instance
    dataset = TokenizedTextDataset(sequences)
    
    # Create a DataLoader instance
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Iterate over the DataLoader
    for batch in dataloader:
        print(batch)