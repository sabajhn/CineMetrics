"""
Module: visualize.py
Description: Generates matplotlib/seaborn figures for the Streamlit app.
             Includes advanced visualizations like WordClouds, Violin plots, and Boxen plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud

class MovieVisualizer:
    """
    Creates visualizations for the analysis.
    """
    def __init__(self):
        # Set a global theme for all plots
        sns.set_theme(style="whitegrid")

    # ... [Previous Plot Methods kept as is: plot_genre_roi, plot_budget_vs_revenue, etc.] ...
    # (I will include the FULL file content below to ensure nothing is lost)

    def plot_genre_roi(self, genre_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(12, 6))
        genre_df = genre_df.sort_values(by='median_roi', ascending=False).head(15)
        sns.barplot(data=genre_df, x='primary_genre', y='median_roi', palette='viridis', hue='primary_genre', legend=False, ax=ax)
        ax.set_title('Return on Investment (ROI) by Genre', fontsize=16)
        ax.set_xlabel('Genre', fontsize=14); ax.set_ylabel('Median ROI', fontsize=14)
        plt.xticks(rotation=55); plt.yticks(fontsize=12)
        return fig

    def plot_budget_vs_revenue(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.regplot(data=df, x='budget', y='revenue', scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'red'}, ax=ax)
        ax.set_title('Budget vs. Revenue (Log Scale)', fontsize=14)
        ax.set_xscale('log'); ax.set_yscale('log')
        return fig

    def plot_yearly_trends(self, yearly_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=yearly_df, x='year', y='avg_budget', marker='o', color='purple', ax=ax)
        ax.set_title('Trend of Average Movie Budgets', fontsize=14)
        return fig

    def plot_seasonal_revenue(self, seasonal_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=seasonal_df, x='month', y='median_revenue', palette='coolwarm', hue='month', legend=False, ax=ax)
        ax.set_title('Revenue by Release Month', fontsize=14); plt.xticks(rotation=45)
        return fig

    def plot_top_studios(self, studio_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=studio_df, y='lead_studio', x='median_revenue', palette='magma', hue='lead_studio', legend=False, ax=ax)
        ax.set_title('Top Studios by Revenue', fontsize=14)
        return fig

    def plot_popularity_vs_rating(self, df: pd.DataFrame):
        df_plot = df.copy(); df_plot['log_popularity'] = np.log1p(df_plot['popularity'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x='log_popularity', y='vote_average', hue='primary_genre', alpha=0.5, s=15, legend=False, ax=ax)
        ax.set_title('Popularity vs. Rating', fontsize=14)
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='mako', hue='feature', legend=False, ax=ax)
        ax.set_title('Feature Importance', fontsize=14)
        return fig

    def plot_confusion_matrix(self, cm, threshold_val):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, xticklabels=['Average', 'Hit'], yticklabels=['Average', 'Hit'])
        ax.set_title(f'Classification Accuracy\n(Hit Threshold: Rating >= {threshold_val:.1f})', fontsize=14)
        return fig

    def plot_decade_pie(self, df: pd.DataFrame):
        df_plot = df.copy().dropna(subset=['year'])
        df_plot['decade'] = (df_plot['year'] // 10 * 10).astype(int).astype(str) + 's'
        counts = df_plot['decade'].value_counts(); total = counts.sum(); threshold = 0.02
        labels = [label if (count/total) > threshold else '' for label, count in counts.items()]
        def autopct_format(pct): return ('%1.1f%%' % pct) if pct > (threshold * 100) else ''
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct=autopct_format, startangle=90, colors=sns.color_palette('pastel'), explode=[0.05 if i == 0 else 0 for i in range(len(counts))], textprops={'fontsize': 10})
        ax.legend(wedges, counts.index, title="Decades", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax.set_title('Movie Releases by Decade', fontsize=16); plt.tight_layout()
        return fig

    def plot_genre_wordcloud(self, df: pd.DataFrame):
        valid_genres = df['genres_list'].dropna()
        all_genres = [str(genre) for sublist in valid_genres if isinstance(sublist, list) for genre in sublist]
        genre_text = " ".join(all_genres)
        if not genre_text.strip(): return None
        wc = WordCloud(background_color="white", width=800, height=400, colormap='magma', max_words=100).generate(genre_text)
        fig, ax = plt.subplots(figsize=(12, 6)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); ax.set_title('Genre Word Cloud', fontsize=16)
        return fig

    def plot_genre_popularity_boxen(self, df: pd.DataFrame):
        df_exploded = df.explode('genres_list')
        top_genres = df_exploded['genres_list'].value_counts().head(10).index
        df_plot = df_exploded[df_exploded['genres_list'].isin(top_genres)]
        df_plot = df_plot[df_plot['popularity'] > 0]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxenplot(data=df_plot, x='genres_list', y='popularity', palette='Spectral', ax=ax)
        ax.set_yscale('log'); ax.set_title('Popularity Distribution (Boxen)', fontsize=16)
        return fig

    def plot_genre_runtime_violin(self, df: pd.DataFrame):
        df_exploded = df.explode('genres_list')
        top_genres = df_exploded['genres_list'].value_counts().head(10).index
        df_plot = df_exploded[df_exploded['genres_list'].isin(top_genres)]
        df_plot = df_plot[(df_plot['runtime'] > 60) & (df_plot['runtime'] < 200)]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(data=df_plot, x='genres_list', y='runtime', palette='Set3', inner='quartile', ax=ax)
        ax.set_title('Runtime Distribution (Violin)', fontsize=16)
        return fig

    def plot_genre_history_stackplot(self, df: pd.DataFrame):
        df_exploded = df.explode('genres_list')
        df_exploded['decade'] = (df_exploded['year'] // 10 * 10).fillna(0).astype(int)
        df_plot = df_exploded[(df_exploded['decade'] >= 1920) & (df_exploded['decade'] <= 2020)]
        top_genres = df_plot['genres_list'].value_counts().head(7).index.tolist()
        df_plot = df_plot[df_plot['genres_list'].isin(top_genres)]
        genre_counts = df_plot.groupby(['decade', 'genres_list']).size().unstack(fill_value=0)
        genre_pcts = genre_counts.div(genre_counts.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.stackplot(genre_pcts.index, genre_pcts.T, labels=genre_pcts.columns, alpha=0.8, cmap='tab10')
        ax.set_title('Genre Evolution (Stackplot)', fontsize=16); ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
        return fig

    def plot_genre_vote_bar(self, df: pd.DataFrame):
        df_exploded = df.explode('genres_list')
        genre_votes = df_exploded.groupby('genres_list')['vote_average'].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=genre_votes.index, y=genre_votes.values, palette='magma', ax=ax)
        ax.set_title('Average Rating by Genre', fontsize=16); ax.set_ylim(4, 8); plt.xticks(rotation=55)
        return fig
