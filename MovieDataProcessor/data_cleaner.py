"""
Module: data_cleaner.py
Description: Encapsulates data ingestion, cleaning, and feature engineering
             within the MovieDataProcessor class.
"""

import os
import ast
import pandas as pd
import numpy as np
from typing import Optional, List, Any

# Configuration Constants
DATA_DIR = "data"
FILENAME = "movies_metadata.csv"
CSV_PATH = os.path.join(DATA_DIR, FILENAME)

class MovieDataProcessor:
    """
    Handles data ingestion, cleaning, and feature engineering for the movie dataset.
    """
    def __init__(self, path: str = CSV_PATH):
        """
        Initializes the processor with the data path.

        Args:
            path (str): The file path to the dataset.
        """
        self.path = path
        self.df: Optional[pd.DataFrame] = None
    
    @staticmethod
    def safe_parse_json(json_str: Any) -> List[Any]:
        """
        Safely parses a stringified JSON list into a Python list.

        Args:
            json_str (Any): The string representation of a list (e.g., "[{'id': 1}]").

        Returns:
            List[Any]: The parsed list, or an empty list if parsing fails.
        """
        if isinstance(json_str, list):
            return json_str
        
        if pd.isna(json_str) or json_str == '':
            return []

        try:
            return ast.literal_eval(str(json_str))
        except (ValueError, SyntaxError):
            return []

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads the movie metadata CSV file and stores it in self.df.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame, or None if loading fails.
        """
        print(f"[INFO] Loading data from: {self.path}...")
        
        cols_to_use = [
            'id', 'title', 'budget', 'revenue', 'release_date', 
            'genres', 'production_companies', 'runtime', 
            'vote_average', 'vote_count', 'popularity'
        ]

        try:
            self.df = pd.read_csv(self.path, usecols=cols_to_use, low_memory=False)
            print(f"[SUCCESS] Data loaded: {len(self.df)} records.")
            return self.df
        except FileNotFoundError:
            print(f"[ERROR] File not found at {self.path}. Please ensure the 'data' directory exists.")
            return None
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while loading data: {e}")
            return None

    def preprocess_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates the data cleaning and feature engineering pipeline.

        Args:
            df_raw (pd.DataFrame): The raw DataFrame returned by load_data.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame.
        """
        print("[INFO] Starting data preprocessing...")
        df = df_raw.copy()
        
        # --- 1. Parse JSON Columns ---
        df['genres_list'] = df['genres'].apply(self.safe_parse_json)
        df['primary_genre'] = df['genres_list'].apply(
            lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan
        )
        
        df['companies_list'] = df['production_companies'].apply(self.safe_parse_json)
        df['lead_studio'] = df['companies_list'].apply(
            lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan
        )

        # --- 2. Numeric Conversion ---
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_count', 'vote_average', 'popularity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['budget'] = df['budget'].replace(0, np.nan)
        df['revenue'] = df['revenue'].replace(0, np.nan)

        # --- 3. Date Handling ---
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year
        df['month'] = df['release_date'].dt.month_name()

        # --- 4. Feature Engineering ---
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = df['profit'] / df['budget']

        # --- 5. Filtering ---
        initial_count = len(df)
        df_clean = df.dropna(subset=['budget', 'revenue', 'primary_genre', 'year', 'month', 'popularity'])
        dropped_count = initial_count - len(df_clean)
        
        print(f"[INFO] Preprocessing complete. {len(df_clean)} valid records remaining (Dropped {dropped_count}).")
        
        return df_clean