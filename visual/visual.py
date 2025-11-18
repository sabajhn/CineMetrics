import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_plot_style():
    """Sets up a consistent, professional style for all plots."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.sans-serif'] = 'Inter'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
def save_plot(fig: plt.Figure, filename: str):
    """Saves the figure to the results directory."""
    save_path = os.path.join("results", filename)
    os.makedirs("results", exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"Plot saved to {save_path}")


def plot_genre_success(genre_summary: pd.DataFrame):
    """
    Creates a grouped bar chart comparing Median ROI and Median Vote Average by Genre.
    """
    setup_plot_style()
    
    # Select the top 10 genres by number of releases for visualization
    top_genres = genre_summary.nlargest(10, 'Total_Releases')
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot 1: Median ROI (Blue bars)
    sns.barplot(
        x='primary_genre', 
        y='Median_ROI', 
        data=top_genres, 
        ax=ax1, 
        color='skyblue', 
        label='Median ROI',
        alpha=0.7
    )
    ax1.set_ylabel("Median ROI (Ratio)", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xlabel("Primary Genre")
    ax1.set_title("Success Metrics by Genre (Top 10 Genres by Release Count)")
    ax1.tick_params(axis='x', rotation=45)
    
    # Create a secondary axis for Median Vote Average
    ax2 = ax1.twinx()
    sns.pointplot(
        x='primary_genre', 
        y='Median_Vote_Avg', 
        data=top_genres, 
        ax=ax2, 
        color='red', 
        linestyles='--', 
        marker='o',
        label='Median Vote Average'
    )
    ax2.set_ylabel("Median Vote Average (0-10)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(4, 8) # Fix y-axis scale for readability of ratings
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    save_plot(fig, "genre_success_comparison.png")
    
    
def plot_budget_vs_revenue(df: pd.DataFrame):
    """
    Creates a scatter plot to show the relationship between budget and revenue.
    Uses log scales for better visibility of highly skewed financial data.
    """
    setup_plot_style()
    
    # Ensure budgets/revenues are non-zero for log scale
    plot_df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Use log scale and linear model fitting from seaborn
    sns.regplot(
        x='budget', 
        y='revenue', 
        data=plot_df, 
        ax=ax, 
        scatter_kws={'alpha': 0.3, 's': 20},
        line_kws={'color': 'red'},
        logx=True
    )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel("Budget (Log Scale, USD)")
    ax.set_ylabel("Revenue (Log Scale, USD)")
    ax.set_title("Relationship between Movie Budget and Box Office Revenue")
    
    save_plot(fig, "budget_vs_revenue_scatter.png")


def plot_runtime_profit(df: pd.DataFrame):
    """
    Creates a boxplot to compare profitability across different runtime categories.
    """
    setup_plot_style()
    
    # Define runtime bins
    bins = [0, 90, 120, np.inf]
    labels = ['Short (<90 min)', 'Standard (90-120 min)', 'Long (>120 min)']
    df['runtime_category'] = pd.cut(df['runtime'], bins=bins, labels=labels, right=False)
    
    # Filter out extreme ROI outliers for visualization clarity (e.g., top 1%)
    max_roi_quantile = df['roi'].quantile(0.99)
    plot_df = df[df['roi'] < max_roi_quantile]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.boxplot(
        x='runtime_category', 
        y='roi', 
        data=plot_df.sort_values(by='runtime_category', ascending=True),
        ax=ax,
        palette="viridis"
    )
    
    ax.set_xlabel("Movie Runtime Category")
    ax.set_ylabel("Return on Investment (ROI)")
    ax.set_title("Profitability (ROI) Distribution by Movie Runtime")
    
    save_plot(fig, "runtime_vs_profitability.png")

def plot_time_trends(time_series_data: pd.DataFrame):
    """
    Creates a line plot to show trends in average budget and revenue over time.
    """
    setup_plot_style()
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot 1: Average Revenue (Blue line)
    sns.lineplot(
        x='year', 
        y='Avg_Revenue', 
        data=time_series_data, 
        ax=ax1, 
        label='Average Revenue', 
        color='green', 
        linewidth=2
    )
    ax1.set_ylabel("Average Revenue (USD)", color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xlabel("Year of Release")
    ax1.set_title("Financial Trends of Released Movies Over Time")

    # Create a secondary axis for Average Budget
    ax2 = ax1.twinx()
    sns.lineplot(
        x='year', 
        y='Avg_Budget', 
        data=time_series_data, 
        ax=ax2, 
        label='Average Budget', 
        color='darkorange', 
        linestyle='--', 
        linewidth=2
    )
    ax2.set_ylabel("Average Budget (USD)", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # Add number of releases to the bottom of the plot as dots
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    sns.lineplot(
        x='year',
        y='Num_Releases',
        data=time_series_data,
        ax=ax3,
        label='Number of Releases',
        color='gray',
        alpha=0.4,
        marker='.',
        linestyle='None'
    )
    ax3.set_ylabel("Number of Releases", color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')
    
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()
    save_plot(fig, "financial_time_trends.png")