"""
Module: model.py
Description: Encapsulates Machine Learning logic within the MoviePredictor class.
             Handles feature preprocessing, model training, and prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

class MoviePredictor:
    """
    Trains a Machine Learning model to predict movie revenue based on features
    like budget, runtime, popularity, and genre.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the predictor with the dataset.
        """
        self.df = df
        self.pipeline = None
        self.metrics = {}
        self.feature_names = []

    def prepare_data(self):
        """
        Selects relevant features and targets for training.
        """
        # Select features that are available before a movie is released
        feature_cols = ['budget', 'runtime', 'popularity', 'primary_genre']
        target_col = 'revenue'

        # Filter out rows where target is missing or 0
        data = self.df.dropna(subset=feature_cols + [target_col])
        data = data[data[target_col] > 0] # Ensure we predict valid revenues

        X = data[feature_cols]
        y = data[target_col]
        
        return X, y

    def train(self):
        """
        Builds a pipeline and trains a Random Forest Regressor.
        """
        X, y = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing for numerical data (fill missing values)
        numeric_features = ['budget', 'runtime', 'popularity']
        numeric_transformer = SimpleImputer(strategy='median')

        # Preprocessing for categorical data (convert Genre to numbers)
        categorical_features = ['primary_genre']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create the full pipeline with a Random Forest Regressor
        # Random Forest is chosen for its robustness and ability to capture non-linear relationships
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Train the model
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        predictions = self.pipeline.predict(X_test)
        self.metrics['mae'] = mean_absolute_error(y_test, predictions)
        self.metrics['r2'] = r2_score(y_test, predictions)
        
        # Save feature names for visualization later
        # (This is a bit tricky with Pipelines, extracting them for the graph)
        onehot_columns = self.pipeline.named_steps['preprocessor'].transformers_[1][1]\
            .named_steps['onehot'].get_feature_names_out(categorical_features)
        self.feature_names = numeric_features + list(onehot_columns)

        return self.metrics

    def predict(self, budget, runtime, popularity, genre):
        """
        Predicts revenue for a single movie instance.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        input_data = pd.DataFrame({
            'budget': [budget],
            'runtime': [runtime],
            'popularity': [popularity],
            'primary_genre': [genre]
        })

        prediction = self.pipeline.predict(input_data)
        return prediction[0]

    def get_feature_importance(self):
        """
        Returns a DataFrame of feature importances.
        """
        if self.pipeline is None:
            return None

        importances = self.pipeline.named_steps['regressor'].feature_importances_
        
        # Match importances with feature names
        if len(self.feature_names) == len(importances):
            df_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            
            # Clean up feature names for better display
            df_importance['feature'] = df_importance['feature'].str.replace('primary_genre_', '')
            return df_importance.head(10)
        
        return None