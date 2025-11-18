import pandas as pd
import numpy as np
import ast
import os

# Define the relative path to the movies metadata CSV
CSV_PATH = os.path.join("data", "movies_metadata.csv")

def load_data(path: str) -> pd.DataFrame or None:
    """
    Loads the movie metadata CSV file.

    Args:
        path (str): The full file path to the movies_metadata.csv file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if loading fails.
    """
    print(f"Loading data from: {path}...")
    try:
        # low_memory=False is necessary for this particular Kaggle dataset 
        # due to mixed data types in some columns.
        df = pd.read_csv(path, low_memory=False)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {path}. Please check your file path and organization.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans, transforms, and prepares the DataFrame for analysis.

    This includes handling stringified JSON, converting data types,
    and calculating key financial metrics.

    Args:
        df (pd.DataFrame): The raw DataFrame.

    Returns:
        pd.DataFrame: The cleaned and processed DataFrame.
    """
    print("Starting data preprocessing...")

    # --- 1. JSON Parsing and Feature Extraction ---
    
    # Define a safe function to extract the first 'name' from stringified JSON lists
    def extract_name(json_str: str, key='name') -> str or np.nan:
        """Extracts the first 'name' value from a stringified JSON list of objects."""
        if pd.isna(json_str):
            return np.nan
        try:
            # Use ast.literal_eval for safe evaluation of stringified lists/dicts
            list_of_dicts = ast.literal_eval(json_str)
            if list_of_dicts and isinstance(list_of_dicts, list):
                # Return the 'name' of the first element
                return list_of_dicts[0].get(key)
            return np.nan
        except (ValueError, SyntaxError):
            # Handle malformed strings, which are present in this dataset
            return np.nan

    # Apply parsing to complex columns
    df['primary_genre'] = df['genres'].apply(lambda x: extract_name(x, 'name'))
    df['production_company'] = df['production_companies'].apply(lambda x: extract_name(x, 'name'))
    df['is_collection'] = df['belongs_to_collection'].apply(lambda x: pd.notna(x))

    # --- 2. Data Type Conversions and Cleaning ---

    # Convert financial columns to numeric, coercing errors
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    
    # Handle zero/missing values in financial columns for analysis
    # Zero budget/revenue often means missing data, so replace with NaN for clean calculation
    df.loc[df['budget'] == 0, 'budget'] = np.nan
    df.loc[df['revenue'] == 0, 'revenue'] = np.nan
    
    # Convert release_date to datetime and extract year
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['year'] = df['release_date'].dt.year

    # Convert vote_average and vote_count to numeric
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

    # --- 3. Feature Engineering (Scientific Computing/Data Manipulation) ---

    # Calculate Profit
    df['profit'] = df['revenue'] - df['budget']

    # Calculate Return on Investment (ROI)
    # ROI = Profit / Budget
    # Use .where() to avoid division by zero (or NaN budget)
    df['roi'] = (df['profit'] / df['budget']).where(df['budget'].notna(), np.nan)

    print("Preprocessing complete. Relevant columns cleaned and new features created.")
    
    # Select only the relevant columns for the final analysis DataFrame
    final_cols = [
        'id', 'title', 'year', 'primary_genre', 'budget', 'revenue', 'profit', 'roi',
        'runtime', 'popularity', 'vote_average', 'vote_count', 'is_collection', 'status'
    ]
    
    # Filter the DataFrame to keep only movies with valid financial and genre data
    df_cleaned = df.dropna(subset=['primary_genre', 'budget', 'revenue', 'vote_count', 'vote_average', 'year'])
    
    # Only keep the final columns (this drops the original 'genres', 'belongs_to_collection', etc.)
    return df_cleaned[final_cols]