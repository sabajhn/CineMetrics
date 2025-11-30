"""
Module: app.py
Description: The main entry point for the Streamlit Web Application.
             It orchestrates Data Processing, Analysis, ML, and Visualization.
"""
import os
import streamlit as st
import ast # Required for list parsing
import pandas as pd
from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH,get_clean_data
from MovieAnalyzer.analysis import MovieAnalyzer
from MovieVisualizer.visualize import MovieVisualizer 

# from models.model_regression import RevenueRegressor
from models.model_classification import SuccessClassifier
# from models.model_recommendation import ContentBasedRecommender 

from models.model_ranking import WeightedRatingRecommender # NEW

st.set_page_config(page_title="CineMetrics: Blockbuster Analytics", page_icon="🎬", layout="wide")

st.title("🎬 CineMetrics: The Blockbuster Blueprint")
st.markdown("Analyze **45,000+ movies** with Data Mining and **Advanced Machine Learning**.")

# --- Data Loading ---
df = get_clean_data()
if df is None:
    st.error(f"❌ Dataset not found at `{CSV_PATH}`.")
    st.stop()

# Helper to restore list structures
def restore_list_cols(df_in):
    df_out = df_in.copy()
    def parse_genres(x):
        try:
            if isinstance(x, str):
                try: x = ast.literal_eval(x)
                except: return []
            if isinstance(x, list):
                if not x: return []
                if isinstance(x[0], dict): return [d.get('name') for d in x if isinstance(d, dict) and 'name' in d]
                if isinstance(x[0], str): return x
            return []
        except: return []

    if 'genres_list' in df_out.columns:
        df_out['genres_list'] = df_out['genres_list'].apply(parse_genres)
    return df_out

df_viz = restore_list_cols(df)

# --- Model Training ---
@st.cache_resource(show_spinner="Training Models...")
def train_models(df):
    clf = SuccessClassifier(df)
    c_met = clf.train()
    return clf, c_met

classifier, clf_metrics = train_models(df)
ranker = WeightedRatingRecommender(df) # Initialize the Ranking model

st.sidebar.success(f"System Ready.\nClassifier Acc: {clf_metrics['accuracy']:.0%}")

analyzer = MovieAnalyzer(df)
visualizer = MovieVisualizer()

# --- Navigation ---
st.sidebar.header("Navigation")
options = st.sidebar.radio("Select Module:", 
    ["Overview", "Financial Analysis", "Genre & Studio", "Data Mining", "Success Classifier", "Top Charts"])

# Shared lists
genres_list = sorted(df['primary_genre'].dropna().unique())
langs_list = sorted(df['original_language'].dropna().unique())
en_index = langs_list.index('en') if 'en' in langs_list else 0
months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
studios_list = sorted(df['lead_studio'].astype(str).unique())

# --- MODULES ---

if options == "Overview":
    st.header("📊 Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(df):,}")
    c2.metric("Avg Budget", f"${df['budget'].mean():,.0f}")
    c3.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}")
    c4.metric("Avg ROI", f"{df['roi'].median():.2f}x")
    
    st.subheader("Raw Data Sample")
    cols_hide = ['genres', 'production_companies', 'genres_list', 'companies_list', 'belongs_to_collection', 'overview', 'tagline']
    st.dataframe(df.drop(columns=cols_hide, errors='ignore').head(10))
    
    st.markdown("### Key Correlations")
    st.dataframe(analyzer.get_correlation_matrix().style.background_gradient(cmap="coolwarm"))

elif options == "Financial Analysis":
    st.header("💰 Financial Analysis")
    t1, t2 = st.tabs(["Correlation", "History"])
    with t1: st.pyplot(visualizer.plot_budget_vs_revenue(df))
    with t2:
        st.pyplot(visualizer.plot_genre_history_stackplot(df_viz))
        c1, c2 = st.columns([2,1])
        with c1: st.pyplot(visualizer.plot_yearly_trends(analyzer.get_yearly_trends()))
        with c2: st.pyplot(visualizer.plot_decade_pie(df_viz))

elif options == "Genre & Studio":
    st.header("🎭 Genre & Studio Analytics")
    if 'genres_list' in df_viz.columns and df_viz['genres_list'].apply(len).sum() > 0:
        st.subheader("Genre Landscape")
        st.pyplot(visualizer.plot_genre_wordcloud(df_viz))
    
    t1, t2 = st.tabs(["Profit & Quality", "Distributions"])
    with t1:
        c1, c2 = st.columns(2)
        with c1: 
            st.pyplot(visualizer.plot_genre_roi(analyzer.get_genre_metrics()))
            st.pyplot(visualizer.plot_genre_vote_bar(df_viz))
        with c2: st.pyplot(visualizer.plot_top_studios(analyzer.get_top_studios()))
    with t2:
        c1, c2 = st.columns(2)
        with c1: st.pyplot(visualizer.plot_genre_popularity_boxen(df_viz))
        with c2: st.pyplot(visualizer.plot_genre_runtime_violin(df_viz))

elif options == "Data Mining":
    st.header("⛏️ Data Mining Deep Dive")
    c1, c2 = st.columns(2)
    with c1: st.pyplot(visualizer.plot_seasonal_revenue(analyzer.get_seasonal_stats()))
    with c2: st.pyplot(visualizer.plot_popularity_vs_rating(df))
    st.subheader("Runtime Sweet Spot"); st.table(analyzer.get_runtime_metrics().head())

elif options == "Success Classifier":
    st.header("🏆 AI Success Classification")
    c1, c2 = st.columns([1, 2])
    with c1:
        b = st.number_input("Budget ($)", 1000, 500000000, 5000000, key='bc')
        r = st.slider("Runtime", 30, 240, 100, key='rc')
        p = st.slider("Popularity", 1.0, 50.0, 5.0, key='pc')
        v = st.number_input("Votes", 0, 15000, 500, key='vc')
        # f = st.checkbox("Franchise?", False, key='fc')
        g = st.selectbox("Genre", genres_list, key='gc')
        l = st.selectbox("Language", langs_list, index=en_index, key='lc')
        m = st.selectbox("Month", months_list, key='mc')
        s = st.selectbox("Studio", studios_list, key='sc')
        if st.button("Classify"):
            prob = classifier.predict_proba(b, r, p, 1 if f else 0, v, g, l, m, s)
            if prob > 0.5: st.balloons(); st.success(f"🌟 **HIT!** ({prob:.1%})")
            else: st.warning(f"📉 **FLOP** ({1-prob:.1%})")
    with c2: st.pyplot(visualizer.plot_confusion_matrix(clf_metrics['confusion_matrix'], classifier.rating_threshold))

# --- MODULE 7: TOP CHARTS (NEW) ---
elif options == "Top Charts":
    st.header("🌟 Top Rated Movies (IMDB Formula)")
    st.markdown("""
    This ranking uses a **Weighted Rating (WR)** formula to balance the rating with the number of votes.
    This prevents movies with very few votes from skimming the top of the list.
    """)
    
    genre_filter = st.selectbox("Filter by Genre (Optional)", ["All Genres"] + genres_list)
    
    if genre_filter == "All Genres":
        top_movies = ranker.get_top_movies()
        st.subheader("🏆 Top 10 Movies of All Time")
    else:
        top_movies = ranker.get_top_movies(genre=genre_filter)
        st.subheader(f"🏆 Top 10 {genre_filter} Movies")
    
    st.table(top_movies)