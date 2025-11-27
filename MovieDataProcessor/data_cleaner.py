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
    Handles data ingestion, cleaning, and feature engineering.
    """
    def __init__(self, path: str = CSV_PATH):
        self.path = path
        self.df: Optional[pd.DataFrame] = None
    
    @staticmethod
    def safe_parse_json(json_str: Any) -> List[Any]:
        """
        Safely parses stringified JSON (e.g., "[{'id': 1, 'name': 'Action'}]").
        Returns a Python list.
        """
        if isinstance(json_str, list):
            return json_str
        if pd.isna(json_str) or json_str == '':
            return []
        try:
            # ast.literal_eval safely evaluates a string containing a Python literal
            return ast.literal_eval(str(json_str))
        except (ValueError, SyntaxError):
            return []

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from CSV with error handling.
        """
        # Only load columns we actually need to save memory
        cols_to_use = [
            'id', 'title', 'budget', 'revenue', 'release_date', 
            'genres', 'production_companies', 'runtime', 
            'vote_average', 'vote_count', 'popularity'
        ]

        try:
            # low_memory=False handles mixed data types in the raw Kaggle file
            self.df = pd.read_csv(self.path, usecols=cols_to_use, low_memory=False)
            return self.df
        except FileNotFoundError:
            # Re-raising allows the App to catch it and show a nice UI error
            raise FileNotFoundError(f"Could not find '{self.path}'. Please check the 'data' folder.")

    def preprocess_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the raw data: parses JSON, converts types, and adds features.
        """
        df = df_raw.copy()
        
        # 1. Parse JSON Columns
        # We extract the first item's 'name' from the list of dictionaries
        df['genres_list'] = df['genres'].apply(self.safe_parse_json)
        df['primary_genre'] = df['genres_list'].apply(
            lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan
        )
        
        df['companies_list'] = df['production_companies'].apply(self.safe_parse_json)
        df['lead_studio'] = df['companies_list'].apply(
            lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan
        )

        # 2. Numeric Conversion
        # Coerce errors to NaN (e.g., if budget is 'unknown')
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_count', 'vote_average', 'popularity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Treat 0 budget/revenue as missing data for accurate financial analysis
        df['budget'] = df['budget'].replace(0, np.nan)
        df['revenue'] = df['revenue'].replace(0, np.nan)

        # 3. Date Handling
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year
        df['month'] = df['release_date'].dt.month_name()

        # 4. Feature Engineering (Scientific Computing)
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = df['profit'] / df['budget']

        # 5. Filtering
        # Drop rows missing critical data for our core analysis
        df_clean = df.dropna(subset=['budget', 'revenue', 'primary_genre', 'year', 'popularity'])
        
        return df_clean