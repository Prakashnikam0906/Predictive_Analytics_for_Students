import pandas as pd

def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
