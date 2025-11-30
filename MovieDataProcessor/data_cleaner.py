"""
Module: data_cleaner.py
Description: Encapsulates data ingestion, cleaning, and feature engineering.
             Now includes the Streamlit-cached loader function.
"""

import os
import ast
import pandas as pd
import numpy as np
import streamlit as st
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
        """Safely parses stringified JSON."""
        if isinstance(json_str, list): return json_str
        if pd.isna(json_str) or json_str == '': return []
        try:
            return ast.literal_eval(str(json_str))
        except (ValueError, SyntaxError):
            return []

    def load_data(self) -> pd.DataFrame:
        """Loads raw data from CSV."""
        cols_to_use = [
            'id', 'title', 'budget', 'revenue', 'release_date', 
            'genres', 'production_companies', 'runtime', 
            'vote_average', 'vote_count', 'popularity',
            'belongs_to_collection', 'original_language',
            'overview', 'tagline' 
        ]
        try:
            self.df = pd.read_csv(self.path, usecols=cols_to_use, low_memory=False)
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find '{self.path}'. Please check the 'data' folder.")

    def preprocess_data(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Cleans the raw data and adds features."""
        df = df_raw.copy()
        
        # 1. Parse JSON
        df['genres_list'] = df['genres'].apply(self.safe_parse_json)
        df['primary_genre'] = df['genres_list'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan)
        
        df['companies_list'] = df['production_companies'].apply(self.safe_parse_json)
        df['lead_studio'] = df['companies_list'].apply(lambda x: x[0]['name'] if isinstance(x, list) and len(x) > 0 else np.nan)

        # 2. Feature Engineering
        df['is_franchise'] = df['belongs_to_collection'].apply(lambda x: 1 if pd.notna(x) and x != '[]' and x != '' else 0)

        # 3. Numeric Conversion
        numeric_cols = ['budget', 'revenue', 'runtime', 'vote_count', 'vote_average', 'popularity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['budget'] = df['budget'].replace(0, np.nan)
        df['revenue'] = df['revenue'].replace(0, np.nan)

        # 4. Date Handling
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['year'] = df['release_date'].dt.year
        df['month'] = df['release_date'].dt.month_name()

        # 5. Profit/ROI
        df['profit'] = df['revenue'] - df['budget']
        df['roi'] = df['profit'] / df['budget']

        # 6. Filtering
        df_clean = df.dropna(subset=['budget', 'revenue', 'primary_genre', 'year', 'popularity', 'original_language'])
        
        return df_clean

# --- STREAMLIT CACHED LOADER (Moved here for better organization) ---
@st.cache_data(show_spinner="Loading and cleaning data...")
def get_clean_data():
    """
    Public function to get the fully processed dataframe.
    Handles the caching logic and unhashable type fix.
    """
    if not os.path.exists(CSV_PATH):
        return None

    processor = MovieDataProcessor(path=CSV_PATH)
    df_raw = processor.load_data()
    df_clean = processor.preprocess_data(df_raw)
    
    # Fix for Streamlit Caching (Convert unhashable types to strings)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                pd.util.hash_pandas_object(df_clean[[col]])
            except TypeError:
                df_clean[col] = df_clean[col].astype(str)
                
    return df_clean