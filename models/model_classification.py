"""
Module: model_classification.py
Description: Contains the SuccessClassifier class using Gradient Boosting
             for predicting Hit vs. Average.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

class SuccessClassifier:
    """
    Gradient Boosting Classifier to predict if a movie is a Critical Hit.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pipeline = None
        self.rating_threshold = 7.0
        self.top_studios = []
        
        # Expanded Features
        self.numeric_features = ['budget', 'runtime', 'popularity', 'is_franchise', 'vote_count']
        self.categorical_features = ['primary_genre', 'original_language', 'month', 'lead_studio']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='median'), self.numeric_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), self.categorical_features)
            ])

    def preprocess_studios(self, data):
        """Groups small studios into 'Other'."""
        top_50 = data['lead_studio'].value_counts().nlargest(50).index.tolist()
        self.top_studios = top_50
        data['lead_studio'] = data['lead_studio'].apply(lambda x: x if x in top_50 else 'Other')
        return data

    def train(self):
        """Trains the Classification model."""
        # 1. Prepare Data
        target_col = 'vote_average'
        data = self.df.dropna(subset=self.numeric_features + self.categorical_features + [target_col])
        
        # Group studios
        data = self.preprocess_studios(data)
        
        # Threshold: Top 25%
        self.rating_threshold = data[target_col].quantile(0.75)
        data['is_hit'] = (data[target_col] >= self.rating_threshold).astype(int)

        X = data[self.numeric_features + self.categorical_features]
        y = data['is_hit']

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Pipeline (Using Gradient Boosting)
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
        ])

        # 4. Fit
        self.pipeline.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = self.pipeline.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'confusion_matrix': confusion_matrix(y_test, preds)
        }
        
        return metrics

    def predict_proba(self, budget, runtime, popularity, vote_count, genre, language, month, studio):
        """Returns the probability of being a Hit."""
        studio_processed = studio if studio in self.top_studios else 'Other'
        
        input_df = pd.DataFrame([[budget, runtime, popularity, vote_count, genre, language, month, studio_processed]], 
                                columns=self.numeric_features + self.categorical_features)
        return self.pipeline.predict_proba(input_df)[0][1]