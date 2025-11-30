# **CineMetrics: The Blockbuster Blueprint**

> **Note:** This project is an advanced data analytics and machine learning dashboard built with **Python** and **Streamlit**.

## **Project Overview**
**CineMetrics** processes a dataset of over 45,000 movies to uncover _hidden trends_, predict <ins>box office revenue</ins>, classify potential hits, and recommend similar movies based on content.

The primary goal of this project is to:
* Provide an **Interactive Dashboard** for high-level metrics (ROI, Budget vs. Revenue).
* Utilize **Machine Learning** to predict financial success.
* Rank movies using a **Weighted Rating Formula**.

---

## **Data Sources**
This project utilizes the "The Movies Dataset" from Kaggle.

> "Data is the new oil." — _Clive Humby_

The data was sourced from the following location:
* **Primary Dataset:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)
    * <sub>Source: Kaggle</sub>
    * <ins>Description:</ins> Metadata on over 45,000 movies.

---

## **Features**

### **1. 📊 Interactive Dashboard**
* **Overview:** High-level metrics including **Total Movies**, **Avg Budget/Revenue**, and <ins>ROI</ins>.
* **Financial Analysis:** Deep dives into Budget vs. Revenue correlations and historical market trends (decades, yearly growth).
* **Genre & Studio Analytics:**
    * Word Clouds for genre visualization.
    * _ROI Analysis_ by Genre.
* **Distributions:** Boxen plots (Popularity) and Violin plots (Runtime).
* **Data Mining:** Seasonality analysis (best month to release?) and Popularity vs. Quality checks.


### **2. 🤖 Advanced Machine Learning**
* **Revenue Predictor (Regression):**
    * Uses **Gradient Boosting** to predict the _exact_ box office revenue.
    * Inputs: Budget, Runtime, Genre, Language, etc.
    * Visualizes **Feature Importance** to show what drives revenue.
* **Success Classifier (Classification):**
    * Uses Gradient Boosting to classify a movie as a **"Hit"** (Critical Success) or **"Average/Flop"**.
    * Includes a <ins>Confusion Matrix</ins> to visualize model performance.

### **3. 🌟 Top Charts (Ranking Algorithm)**
* Implements the **IMDB Weighted Rating Formula** to rank movies fairly.
* Balances their average rating with the number of votes they received.
* Allows filtering by **Genre** (e.g., "Top 10 Horror Movies").

### **4. 🔮 Prediction Analysis**
* Visualizes the relationships that drive the Machine Learning models.
* Includes **Scatter plots** of Features vs. Targets, Joint Distributions, and Correlation Heatmaps.

---

## **Project Structure**
The project follows a professional Object-Oriented Programming (OOP) structure for modularity and scalability.

```text
CineMetrics/
├── data/
│   └── data_loader.py       # Handles data loading & caching
├── models/
│   ├── model_regression.py  # Revenue prediction logic (Gradient Boosting)
│   ├── model_classification.py # Success classification logic
│   └── model_ranking.py     # Weighted Rating (Top Charts) logic
├── plots/
│   ├── prediction_plots.py  # Plots for ML analysis (Scatter, Joint, Heatmap)
│   └── recommendation_plots.py # Similarity score visualization
├── MovieAnalyzer/
│   └── analysis.py          # Statistical aggregations (GroupBys, Trends)
├── MovieDataProcessor/
│   └── data_cleaner.py      # Data cleaning, JSON parsing, Feature Engineering
├── MovieVisualizer/
│   └── visualize.py         # Core visualization logic (Seaborn/Matplotlib)
├── app.py                   # Main Streamlit application entry point
├── requirements.txt         # List of dependencies
├── .gitignore               # Files to exclude from Git
└── README.md                # Project documentation

# Setup & Installation

Follow these instructions to set up the project on your local machine.

---

## 1. Prerequisites
Ensure you have **Python 3.8+** installed.

---

## 2. Install Dependencies
Run the following command to install all required libraries (Pandas, Scikit-Learn, Streamlit, Seaborn, etc.):

```bash
pip install -r requirements.txt

