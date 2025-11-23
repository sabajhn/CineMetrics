"""
Module: visualize.py
Description: Encapsulates plot generation and saving within the MovieVisualizer class.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Configuration Constant
RESULTS_DIR = "results"

class MovieVisualizer:
    """
    Generates and saves analytical plots using Matplotlib and Seaborn.
    """
    def __init__(self):
        """Initializes the visualizer and ensures the results directory exists."""
        self.results_dir = RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        sns.set_theme(style="whitegrid") # Set style once
        print(f"[INFO] Visualizations will be saved to: {self.results_dir}")

    def save_plot(self, filename: str):
        """Helper function to save and close plots."""
        path = os.path.join(self.results_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=300) 
        plt.close()
        print(f"[PLOT] Saved: {path}")

    def plot_genre_roi(self, genre_df: pd.DataFrame):
        """Generates a bar chart for Median ROI by Genre."""
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=genre_df.head(10), 
            x='primary_genre', 
            y='median_roi', 
            palette='viridis', 
            hue='primary_genre', 
            legend=False
        )
        plt.title('Top 10 Genres by Median ROI (Return on Investment)', fontsize=14)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Median ROI', fontsize=12)
        plt.xticks(rotation=45)
        self.save_plot('genre_roi.png')

    def plot_budget_vs_revenue(self, df: pd.DataFrame):
        """Generates a regression plot for Budget vs. Revenue."""
        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=df, 
            x='budget', 
            y='revenue', 
            scatter_kws={'alpha': 0.3, 's': 10}, 
            line_kws={'color': 'red'}
        )
        plt.title('Correlation: Budget vs. Revenue (Log Scale)', fontsize=14)
        plt.xlabel('Budget (USD, Log Scale)', fontsize=12)
        plt.ylabel('Revenue (USD, Log Scale)', fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        self.save_plot('budget_vs_revenue.png')

    def plot_yearly_trends(self, yearly_df: pd.DataFrame):
        """Generates a line chart for budget trends over time."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=yearly_df, 
            x='year', 
            y='avg_budget', 
            marker='o', 
            color='purple'
        )
        plt.title('Trend of Average Movie Budgets (1980-Present)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average Budget (USD)', fontsize=12)
        self.save_plot('yearly_budget_trend.png')

    def plot_seasonal_revenue(self, seasonal_df: pd.DataFrame):
        """Generates a bar chart for revenue seasonality."""
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=seasonal_df, 
            x='month', 
            y='median_revenue', 
            palette='coolwarm', 
            hue='month', 
            legend=False
        )
        plt.title('Seasonality: Median Revenue by Release Month', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Median Revenue (USD)', fontsize=12)
        plt.xticks(rotation=45)
        self.save_plot('seasonal_revenue.png')

    def plot_top_studios(self, studio_df: pd.DataFrame):
        """Generates a horizontal bar chart for top production studios."""
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=studio_df, 
            y='lead_studio', 
            x='median_revenue', 
            palette='magma', 
            hue='lead_studio', 
            legend=False
        )
        plt.title('Top 10 Major Studios by Median Box Office Revenue', fontsize=14)
        plt.xlabel('Median Revenue (USD)', fontsize=12)
        plt.ylabel('Production Company', fontsize=12)
        self.save_plot('top_studios.png')

    def plot_runtime_vs_vote(self, runtime_df: pd.DataFrame):
        """Generates a bar chart showing average vote by runtime bin."""
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=runtime_df, 
            x='runtime_bin', 
            y='avg_vote', 
            palette='cividis', 
            hue='runtime_bin', 
            legend=False
        )
        plt.title('Average Rating by Movie Runtime', fontsize=14)
        plt.xlabel('Runtime Bin (Minutes)', fontsize=12)
        plt.ylabel('Average Vote (TMDB)', fontsize=12)
        self.save_plot('runtime_vs_vote.png')
        
    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        """Generates a scatter plot to show the relationship between popularity and rating."""
        
        # Calculate the log-transformed popularity column for plotting
        df['log_popularity'] = np.log1p(df['popularity'])
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='log_popularity', 
            y='vote_average', 
            hue='primary_genre', 
            alpha=0.5, 
            s=15,
            legend=False
        )
        plt.title('Popularity vs. Critic Rating (Vote Average)', fontsize=14)
        plt.xlabel('Log(1 + Popularity Score)', fontsize=12)
        plt.ylabel('Vote Average (1-10)', fontsize=12)
        self.save_plot('popularity_vs_rating.png')