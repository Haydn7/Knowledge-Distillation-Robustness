from datasets import Dataset
import pandas as pd

class CSVDataset(Dataset):
    """ Interface for pandas dataframe into a Dataset"""

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): Path to the CSV file.
            transform (callable, optional): Optional transform to be applied to a sample.
        """
        self.df = pd.read_csv(file_path)  # Load the CSV file into a DataFrame
        self.transform = transform

    @property
    def column_names(self):
        return self.df.columns

    def __len__(self):
        return len(self.df)  # Number of samples in the dataset

    def __getitem__(self, idx):
        sample = self.df.iloc[idx].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample
