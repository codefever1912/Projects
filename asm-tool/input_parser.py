import pandas as pd
import os

class InputParser:
    def __init__(self, input_file=None):
        self.input_file = input_file

    def load_domains(self):
        if self.input_file:
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file {self.input_file} not found.")
            df = pd.read_csv(self.input_file)
            if 'domain' not in df.columns:
                raise ValueError(f"CSV file must contain a 'domain' column.")
            return df['domain'].tolist()
        else:
            raise ValueError("No input provided. Please provide a valid CSV or domain input.")
