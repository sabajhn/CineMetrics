"""
Module: visualize.py
Description: Generates matplotlib/seaborn figures for the Streamlit app.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class MovieVisualizer:
    """
    Creates visualizations for the analysis.
    """
    def __init__(self):
        # Set a global theme for all plots
        sns.set_theme(style="whitegrid")

    def plot_genre_roi(self, genre_df: pd.DataFrame):
        """Returns a bar chart figure for Median ROI by Genre."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=genre_df.head(10), 
            x='primary_genre', 
            y='median_roi', 
            palette='viridis', 
            hue='primary_genre', 
            legend=False,
            ax=ax
        )
        ax.set_title('Top 10 Genres by Return on Investment (ROI)', fontsize=14)
        ax.set_xlabel('Genre', fontsize=12)
        ax.set_ylabel('Median ROI (Profit/Budget)', fontsize=12)
        plt.xticks(rotation=45)
        return fig

    def plot_budget_vs_revenue(self, df: pd.DataFrame):
        """Returns a scatter plot figure with regression line."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.regplot(
            data=df, 
            x='budget', 
            y='revenue', 
            scatter_kws={'alpha': 0.3, 's': 10}, 
            line_kws={'color': 'red'},
            ax=ax
        )
        ax.set_title('Correlation: Budget vs. Revenue (Log Scale)', fontsize=14)
        ax.set_xlabel('Budget (USD)', fontsize=12)
        ax.set_ylabel('Revenue (USD)', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        return fig

    def plot_yearly_trends(self, yearly_df: pd.DataFrame):
        """Returns a line chart figure for budget trends."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=yearly_df, 
            x='year', 
            y='avg_budget', 
            marker='o', 
            color='purple',
            ax=ax
        )
        ax.set_title('Trend of Average Movie Budgets (1980-Present)', fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Average Budget (USD)', fontsize=12)
        return fig

    def plot_seasonal_revenue(self, seasonal_df: pd.DataFrame):
        """Returns a bar chart figure for seasonality."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=seasonal_df, 
            x='month', 
            y='median_revenue', 
            palette='coolwarm', 
            hue='month', 
            legend=False,
            ax=ax
        )
        ax.set_title('Blockbuster Season: Revenue by Release Month', fontsize=14)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Median Revenue (USD)', fontsize=12)
        plt.xticks(rotation=45)
        return fig

    def plot_top_studios(self, studio_df: pd.DataFrame):
        """Returns a horizontal bar chart figure for top studios."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.barplot(
            data=studio_df, 
            y='lead_studio', 
            x='median_revenue', 
            palette='magma', 
            hue='lead_studio', 
            legend=False,
            ax=ax
        )
        ax.set_title('Top 10 Major Studios by Median Box Office', fontsize=14)
        ax.set_xlabel('Median Revenue (USD)', fontsize=12)
        ax.set_ylabel('Production Company', fontsize=12)
        return fig

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        """Returns a scatter plot of Popularity vs Rating."""
        # Log transform for better visualization
        df_plot = df.copy()
        df_plot['log_popularity'] = np.log1p(df_plot['popularity'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_plot, 
            x='log_popularity', 
            y='vote_average', 
            hue='primary_genre',
            alpha=0.5, 
            s=15,
            legend=False, # Legend removed for clarity in dense plots
            ax=ax
        )
        ax.set_title('Do Popular Movies get Better Ratings?', fontsize=14)
        ax.set_xlabel('Log(Popularity Score)', fontsize=12)
        ax.set_ylabel('Vote Average (0-10)', fontsize=12)
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        """Returns a bar chart of the Machine Learning Feature Importance."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            data=importance_df,
            x='importance',
            y='feature',
            palette='mako',
            hue='feature',
            legend=False,
            ax=ax
        )
        ax.set_title('What drives Revenue? (Model Feature Importance)', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        return fig