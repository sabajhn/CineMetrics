"""
Module: main.py
Description: Entry point for the CineMetrics analysis pipeline. This script
             orchestrates the MovieDataProcessor, MovieAnalyzer, and MovieVisualizer classes.
"""

import os
import sys
import pandas as pd

# --- Import Handling ---
# Ensures that local modules can be imported even if running from a different context
try:
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    
    from MovieDataProcessor.data_cleaner import MovieDataProcessor, CSV_PATH
    from MovieAnalyzer.analysis import MovieAnalyzer
    from MovieVisualizer.visualize import MovieVisualizer 
    
    print("[SYSTEM] Modules loaded successfully.\n")

except ImportError as e:
    print(f"\n[CRITICAL ERROR] Module import failed: {e}")
    print("Ensure all Python files are in the same directory.")
    sys.exit(1)

def main():
    print("==========================================")
    print("   CineMetrics: Blockbuster Analytics     ")
    print("==========================================\n")

    # --- 1. Data Processing ---
    
    # Initialize processor
    processor = MovieDataProcessor(path=CSV_PATH)
    
    # Check for file existence before loading
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Dataset not found at: {CSV_PATH}")
        print("Please create a 'data' folder and place 'movies_metadata.csv' inside it.")
        return

    # Load Data
    df_raw = processor.load_data() 
    if df_raw is None:
        return

    # Preprocess Data
    df_clean = processor.preprocess_data(df_raw)
    if df_clean.empty:
        print("[ERROR] Dataset is empty after preprocessing. Check input file integrity.")
        return
    
    # --- 2. Statistical Analysis ---
    
    print("\n[INFO] Performing statistical analysis...")
    analyzer = MovieAnalyzer(df_clean)
    
    # Run all analysis methods
    genre_metrics = analyzer.get_genre_metrics()
    yearly_trends = analyzer.get_yearly_trends()
    correlations = analyzer.get_correlation_matrix()
    seasonal_stats = analyzer.get_seasonal_stats()
    top_studios = analyzer.get_top_studios()
    runtime_metrics = analyzer.get_runtime_metrics()
    
    # --- Console Insights ---
    print("\n--- 💰 Top 5 Most Profitable Genres (Median ROI) ---")
    print(genre_metrics[['primary_genre', 'median_roi', 'count']].head(5).to_string(index=False))
    
    print("\n--- 📅 Best Month for Revenue ---")
    best_month = seasonal_stats.sort_values(by='median_revenue', ascending=False).iloc[0]
    print(f"Month: {best_month['month']}, Median Revenue: ${best_month['median_revenue']:,.2f}")

    print("\n--- 🎬 Top Studio by Revenue ---")
    best_studio = top_studios.iloc[0]
    print(f"Studio: {best_studio['lead_studio']}, Median Revenue: ${best_studio['median_revenue']:,.2f}")

    print("\n--- ⏱️ Average Rating by Runtime Bin ---")
    print(runtime_metrics[['runtime_bin', 'avg_vote', 'median_revenue']].to_string(index=False))

    print("\n--- 🔗 Key Correlations ---")
    print(f"Budget vs Revenue: {correlations.loc['budget', 'revenue']:.4f}")
    print(f"Runtime vs Revenue: {correlations.loc['runtime', 'revenue']:.4f}")

    # --- 3. Visualization ---
    
    print("\n[INFO] Generating visualizations...")
    visualizer = MovieVisualizer()
    
    try:
        visualizer.plot_genre_roi(genre_metrics)
        visualizer.plot_budget_vs_revenue(df_clean)
        visualizer.plot_yearly_trends(yearly_trends)
        visualizer.plot_seasonal_revenue(seasonal_stats)
        visualizer.plot_top_studios(top_studios)
        visualizer.plot_runtime_vs_vote(runtime_metrics)
        visualizer.plot_popularity_vs_rating(df_clean)
        print("\n[SUCCESS] Analysis complete! Results saved to the 'results' directory.")
    except Exception as e:
        print(f"[ERROR] Visualization failed: {e}")

if __name__ == "__main__":
    main()