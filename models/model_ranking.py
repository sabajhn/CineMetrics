"""
Module: model_ranking.py
Description: Implements the 'Simple Recommender' using IMDB's Weighted Rating formula.
             This ranks movies based on a balance of rating and popularity.
"""

import pandas as pd
import numpy as np

class WeightedRatingRecommender:
    """
    Generates top movie charts using the IMDB Weighted Rating formula.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Calculate C (Mean vote across the whole dataset)
        self.C = self.df['vote_average'].mean()
        
        # Calculate m (Minimum votes required to be listed)
        # We use the 90th percentile as our cutoff, meaning a movie must have more votes
        # than 90% of the movies in the list to be considered.
        self.m = self.df['vote_count'].quantile(0.90)

    def weighted_rating(self, x, m=None, C=None):
        """
        Calculates the weighted rating for a single row.
        Formula: (v/(v+m) * R) + (m/(m+v) * C)
        """
        if m is None: m = self.m
        if C is None: C = self.C
        
        v = x['vote_count']
        R = x['vote_average']
        
        return (v / (v + m) * R) + (m / (m + v) * C)

    def get_top_movies(self, genre=None, n=10):
        """
        Returns the top N movies, optionally filtered by genre.
        """
        # Filter for movies that qualify for the chart
        q_movies = self.df.copy().loc[self.df['vote_count'] >= self.m]
        
        # If a genre is specified, filter for that genre
        if genre:
            # Handle the list-like genre column or string
            # We assume 'primary_genre' is a single string for simplicity in this project,
            # but we can also check the list column if needed.
            # For now, we check 'primary_genre' which is cleaner.
            q_movies = q_movies[q_movies['primary_genre'] == genre]
            
            # Recalculate m and C for the specific genre subset if strict accuracy is needed,
            # but typically global C is used for comparison. We'll stick to global C/m for consistency.

        # Calculate Score
        q_movies['score'] = q_movies.apply(self.weighted_rating, axis=1)
        
        # Sort
        q_movies = q_movies.sort_values('score', ascending=False)
        
        # Return top N with relevant columns
        return q_movies[['title', 'primary_genre', 'vote_average', 'vote_count', 'score']].head(n)