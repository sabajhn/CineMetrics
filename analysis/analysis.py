import pandas as pd
import numpy as np

def calculate_success_metrics(df: pd.DataFrame) -> dict:
    """
    Calculates key success metrics (profitability and rating) grouped by primary genre.

    Args:
        df (pd.DataFrame): The cleaned movie DataFrame.

    Returns:
        dict: A dictionary containing the summary DataFrame and a correlation matrix.
    """
    print("Performing scientific computing and statistical analysis...")
    
    # --- Genre Success Analysis ---
    
    # Group by primary genre and calculate key aggregate statistics
    genre_summary = df.groupby('primary_genre').agg(
        Total_Releases=('id', 'count'),
        Median_Budget=('budget', 'median'),
        Median_Revenue=('revenue', 'median'),
        Median_ROI=('roi', 'median'),
        Median_Vote_Avg=('vote_average', 'median'),
        Total_Vote_Count=('vote_count', 'sum')
    ).reset_index()
    
    # Filter for genres with a significant number of releases (e.g., at least 50)
    genre_summary = genre_summary[genre_summary['Total_Releases'] >= 50].sort_values(
        by='Median_ROI', ascending=False
    )
    
    print(f"Summary generated for {len(genre_summary)} significant genres.")

    # --- Correlation Analysis ---

    # Select numerical columns for correlation calculation
    correlation_data = df[['budget', 'revenue', 'profit', 'runtime', 'vote_average', 'vote_count', 'popularity']]
    
    # Calculate Pearson correlation matrix
    correlation_matrix = correlation_data.corr(numeric_only=True)
    
    print("Correlation matrix calculated.")

    # --- Time Series Analysis ---
    
    # Calculate average financial metrics and releases per year
    time_series_data = df.groupby('year').agg(
        Avg_Budget=('budget', 'mean'),
        Avg_Revenue=('revenue', 'mean'),
        Num_Releases=('id', 'count')
    ).reset_index()
    
    # Filter for years with a minimum number of releases to avoid outliers
    time_series_data = time_series_data[time_series_data['Num_Releases'] >= 10].sort_values(by='year')
    
    return {
        'genre_summary': genre_summary,
        'correlation_matrix': correlation_matrix,
        'time_series_data': time_series_data
    }