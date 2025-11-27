"""
Module: app.py
Description: The main entry point for the Streamlit Web Application.
             It orchestrates Data Processing, Analysis, ML, and Visualization.
"""

import streamlit as st
import pandas as pd
from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH
from MovieAnalyzer.analysis import MovieAnalyzer
from MovieVisualizer.visualize import MovieVisualizer 

from models.model import MoviePredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="CineMetrics: Blockbuster Analytics",
    page_icon="🎬",
    layout="wide"
)

# --- Title & Intro ---
st.title("🎬 CineMetrics: The Blockbuster Blueprint")
st.markdown("""
This interactive dashboard explores the factors that determine cinematic success. 
We analyze **45,000+ movies** to uncover trends and use **Machine Learning** to predict future hits.
""")

# --- 1. Data Loading (Cached) ---
@st.cache_data
def load_and_process_data():
    processor = MovieDataProcessor(path=CSV_PATH)
    df_raw = processor.load_data()
    df_clean = processor.preprocess_data(df_raw)
    
    # CRITICAL FIX for Streamlit caching (TypeError: unhashable type: 'list'):
    # Drop any temporary columns created during preprocessing that contain lists (unhashable)
    # The columns 'genres_list' and 'companies_list' from data_cleaner are the likely culprits.
    unhashable_cols = [col for col in df_clean.columns if df_clean[col].apply(lambda x: isinstance(x, list)).any()]
    df_final = df_clean.drop(columns=unhashable_cols, errors='ignore')
    
    return df_final

# --- 2. Model Training (Cached) ---
@st.cache_resource
def train_model(df):
    predictor = MoviePredictor(df)
    metrics = predictor.train()
    return predictor, metrics

# Check data existence
import os
if not os.path.exists(CSV_PATH):
    st.error(f"❌ Critical Error: Dataset not found at `{CSV_PATH}`.")
    st.info("Please create a 'data' folder and put 'movies_metadata.csv' inside it.")
    st.stop()

# Execute Loading
with st.spinner('Loading data and training AI model...'):
    try:
        df = load_and_process_data()
        predictor, metrics = train_model(df)
        st.success(f"System ready! Analyzed {len(df):,} movies. ML Model Accuracy (R²): {metrics['r2']:.2f}")
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

# --- Initialize Classes ---
analyzer = MovieAnalyzer(df)
visualizer = MovieVisualizer()

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Choose Analysis Module:", 
    ["Overview & Stats", "Financial Analysis", "Genre & Studio Insights", "Data Mining Deep Dive", "🔮 ML Revenue Predictor"]
)

# --- MODULE 1: Overview & Stats ---
if options == "Overview & Stats":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Movies", f"{len(df):,}")
    col2.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    col3.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    col4.metric("Avg ROI", f"{df['roi'].median():.2f}x")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Key Correlations")
    corr_matrix = analyzer.get_correlation_matrix()
    st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm"))

# --- MODULE 2: Financial Analysis ---
elif options == "Financial Analysis":
    st.header("💰 Financial Analysis")
    
    st.subheader("Budget vs. Revenue")
    st.markdown("Does throwing money at a movie guarantee success?")
    fig_budget = visualizer.plot_budget_vs_revenue(df)
    st.pyplot(fig_budget)
    
    st.subheader("Historical Trends")
    st.markdown("How have movie budgets changed since 1980?")
    yearly_data = analyzer.get_yearly_trends()
    fig_trend = visualizer.plot_yearly_trends(yearly_data)
    st.pyplot(fig_trend)

# --- MODULE 3: Genre & Studio Insights ---
elif options == "Genre & Studio Insights":
    st.header("🎭 Genre & Studio Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Profitable Genres")
        genre_data = analyzer.get_genre_metrics()
        fig_genre = visualizer.plot_genre_roi(genre_data)
        st.pyplot(fig_genre)
        with st.expander("View Genre Data"):
            st.dataframe(genre_data)

    with col2:
        st.subheader("Top Studios (Revenue)")
        studio_data = analyzer.get_top_studios()
        fig_studio = visualizer.plot_top_studios(studio_data)
        st.pyplot(fig_studio)

# --- MODULE 4: Data Mining Deep Dive ---
elif options == "Data Mining Deep Dive":
    st.header("⛏️ Data Mining: Hidden Patterns")
    
    st.subheader("1. The 'Blockbuster Season'")
    st.markdown("Is there a specific month where movies make the most money?")
    seasonal_data = analyzer.get_seasonal_stats()
    fig_season = visualizer.plot_seasonal_revenue(seasonal_data)
    st.pyplot(fig_season)
    
    st.subheader("2. Popularity vs. Quality")
    st.markdown("Are popular movies actually rated higher by critics?")
    fig_pop = visualizer.plot_popularity_vs_rating(df)
    st.pyplot(fig_pop)
    
    st.subheader("3. The Runtime Sweet Spot")
    st.markdown("Do audiences prefer specific movie lengths?")
    runtime_data = analyzer.get_runtime_metrics()
    st.table(runtime_data)

# --- MODULE 5: Machine Learning (BONUS) ---
elif options == "🔮 ML Revenue Predictor":
    st.header("🔮 AI Revenue Prediction")
    st.markdown("""
    Use our trained **Random Forest Machine Learning model** to predict the Box Office Revenue for a hypothetical movie.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Movie Parameters")
        
        # User Inputs
        budget_input = st.number_input("Budget (USD)", min_value=1000, value=10000000, step=1000000)
        runtime_input = st.slider("Runtime (Minutes)", 30, 240, 120)
        popularity_input = st.slider("Expected Popularity Score", 1.0, 50.0, 10.0)
        
        # Get list of unique genres from data for the dropdown
        unique_genres = sorted(df['primary_genre'].dropna().unique())
        genre_input = st.selectbox("Primary Genre", unique_genres)
        
        predict_btn = st.button("Predict Box Office Revenue", type="primary")

    with col2:
        if predict_btn:
            prediction = predictor.predict(budget_input, runtime_input, popularity_input, genre_input)
            
            st.success(f"💰 Predicted Revenue: **${prediction:,.2f}**")
            
            # Simple ROI calc for the prediction
            predicted_roi = (prediction - budget_input) / budget_input
            if predicted_roi > 0:
                st.metric("Estimated ROI", f"+{predicted_roi:.2f}x", delta_color="normal")
            else:
                st.metric("Estimated ROI", f"{predicted_roi:.2f}x", delta_color="inverse")

        st.markdown("---")
        st.subheader("What drives the prediction?")
        st.markdown("The chart below shows which features the AI finds most important.")
        
        feature_importance = predictor.get_feature_importance()
        if feature_importance is not None:
            fig_imp = visualizer.plot_feature_importance(feature_importance)
            st.pyplot(fig_imp)