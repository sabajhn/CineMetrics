"""
Module: analysis.py
Description: Encapsulates statistical calculations within the MovieAnalyzer class.
"""

import pandas as pd
import numpy as np

class MovieAnalyzer:
    """
    Performs statistical analysis and aggregations.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_genre_metrics(self, min_movie_count: int = 50) -> pd.DataFrame:
        """
        Calculates median Budget, Revenue, and ROI by Genre.
        Filters out niche genres with few movies to ensure statistical validity.
        """
        genre_stats = self.df.groupby('primary_genre').agg(
            count=('id', 'count'),
            median_budget=('budget', 'median'),
            median_revenue=('revenue', 'median'),
            median_roi=('roi', 'median'),
            avg_vote=('vote_average', 'mean')
        ).reset_index()
        
        return genre_stats[genre_stats['count'] >= min_movie_count].sort_values(by='median_roi', ascending=False)

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Computes Pearson correlation between numerical columns.
        """
        cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'roi']
        return self.df[cols].corr(numeric_only=True)

    def get_yearly_trends(self, start_year: int = 1980) -> pd.DataFrame:
        """
        Aggregates financials by year to visualize industry growth.
        """
        yearly_stats = self.df.groupby('year').agg(
            total_revenue=('revenue', 'sum'),
            avg_budget=('budget', 'mean'),
            movie_count=('id', 'count')
        ).reset_index()
        
        return yearly_stats[yearly_stats['year'] >= start_year]

    def get_seasonal_stats(self) -> pd.DataFrame:
        """
        Analyzes success based on release month (Seasonality).
        """
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        seasonal_stats = self.df.groupby('month', observed=False).agg( 
            median_revenue=('revenue', 'median'),
            median_roi=('roi', 'median'),
            count=('id', 'count')
        ).reset_index()
        
        # Sort chronologically
        seasonal_stats['month'] = pd.Categorical(seasonal_stats['month'], categories=month_order, ordered=True)
        return seasonal_stats.sort_values('month')

    def get_top_studios(self, min_movie_count: int = 20) -> pd.DataFrame:
        """
        Finds the most successful studios (highest median revenue).
        """
        studio_stats = self.df.groupby('lead_studio').agg(
            median_revenue=('revenue', 'median'),
            median_roi=('roi', 'median'),
            count=('id', 'count')
        ).reset_index()
        
        # Filter for major studios
        filtered = studio_stats[studio_stats['count'] >= min_movie_count]
        return filtered.sort_values(by='median_revenue', ascending=False).head(10)

    def get_runtime_metrics(self) -> pd.DataFrame:
        """
        Bins movies by length to find the 'sweet spot' for ratings.
        """
        # Filter reasonable range
        df_filtered = self.df[(self.df['runtime'] >= 60) & (self.df['runtime'] <= 240)].copy()

        bins = [0, 90, 120, 150, 240, np.inf]
        labels = ['< 90m', '90-120m', '120-150m', '150-240m', '> 240m']

        df_filtered['runtime_bin'] = pd.cut(df_filtered['runtime'], bins=bins, labels=labels, right=False)

        runtime_stats = df_filtered.groupby('runtime_bin', observed=True).agg(
            count=('id', 'count'),
            median_revenue=('revenue', 'median'),
            avg_vote=('vote_average', 'mean')
        ).reset_index()

        return runtime_stats.dropna()