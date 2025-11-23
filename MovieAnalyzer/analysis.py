"""
Module: analysis.py
Description: Encapsulates statistical analysis and data aggregation
             within the MovieAnalyzer class.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Any

class MovieAnalyzer:
    """
    Performs statistical analysis and data aggregation on a cleaned movie DataFrame.
    """
    def __init__(self, df: pd.DataFrame):
        """Initializes the analyzer with the cleaned DataFrame."""
        self.df = df

    def get_genre_metrics(self, min_movie_count: int = 50) -> pd.DataFrame:
        """
        Calculates success metrics (Budget, Revenue, ROI) grouped by Genre.

        Args:
            min_movie_count (int): Minimum number of movies required to include a genre.

        Returns:
            pd.DataFrame: Aggregated metrics sorted by Median ROI.
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
        Computes the Pearson correlation coefficient between numerical features.

        Returns:
            pd.DataFrame: A correlation matrix.
        """
        cols = ['budget', 'revenue', 'runtime', 'vote_average', 'vote_count', 'popularity', 'roi']
        return self.df[cols].corr(numeric_only=True)

    def get_yearly_trends(self, start_year: int = 1980) -> pd.DataFrame:
        """
        Aggregates financial data by year to analyze industry trends.

        Args:
            start_year (int): The starting year for the analysis.

        Returns:
            pd.DataFrame: Yearly aggregated data.
        """
        yearly_stats = self.df.groupby('year').agg(
            total_revenue=('revenue', 'sum'),
            avg_budget=('budget', 'mean'),
            movie_count=('id', 'count')
        ).reset_index()
        
        return yearly_stats[yearly_stats['year'] >= start_year]

    def get_seasonal_stats(self) -> pd.DataFrame:
        """
        Analyzes revenue and ROI based on the month of release (Seasonality).

        Returns:
            pd.DataFrame: Aggregated stats sorted chronologically by month.
        """
        month_order = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        # FutureWarning suppression, as using observed=True is the intended behavior with Categorical data
        seasonal_stats = self.df.groupby('month', observed=True).agg( 
            median_revenue=('revenue', 'median'),
            median_roi=('roi', 'median'),
            count=('id', 'count')
        ).reset_index()
        
        # Enforce chronological order
        seasonal_stats['month'] = pd.Categorical(seasonal_stats['month'], categories=month_order, ordered=True)
        return seasonal_stats.sort_values('month')

    def get_top_studios(self, min_movie_count: int = 20) -> pd.DataFrame:
        """
        Identifies top-performing production studios based on median revenue.

        Args:
            min_movie_count (int): Filter to exclude small or one-hit studios.

        Returns:
            pd.DataFrame: Top 10 studios sorted by revenue.
        """
        studio_stats = self.df.groupby('lead_studio').agg(
            median_revenue=('revenue', 'median'),
            median_budget=('budget', 'median'),
            median_roi=('roi', 'median'),
            count=('id', 'count')
        ).reset_index()
        
        filtered_studios = studio_stats[studio_stats['count'] >= min_movie_count]
        return filtered_studios.sort_values(by='median_revenue', ascending=False).head(10)

    def get_runtime_metrics(self) -> pd.DataFrame:
        """
        Analyzes movie performance metrics across different runtime bins.

        Returns:
            pd.DataFrame: Aggregated runtime metrics, sorted by average vote.
        """
        # Filter out very short films (like shorts) and extremely long films
        df_filtered = self.df[(self.df['runtime'] >= 60) & (self.df['runtime'] <= 240)].copy()

        # Define runtime bins
        bins = [0, 90, 120, 150, 240, np.inf]
        labels = ['< 90 min', '90-120 min', '120-150 min', '150-240 min', '> 240 min']

        df_filtered['runtime_bin'] = pd.cut(df_filtered['runtime'], bins=bins, labels=labels, right=False)

        # Suppressing FutureWarning by using observed=True, as intended for categorical bins
        runtime_stats = df_filtered.groupby('runtime_bin', observed=True).agg(
            count=('id', 'count'),
            median_revenue=('revenue', 'median'),
            avg_vote=('vote_average', 'mean')
        ).reset_index()

        # Drop the "over 240 min" bin if it has NaN votes/revenue (due to inf bin boundary)
        return runtime_stats.dropna(subset=['runtime_bin', 'avg_vote']).sort_values(by='avg_vote', ascending=False)