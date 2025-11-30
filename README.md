# **CineMetrics: The Blockbuster Blueprint**

> **Note:** This project is an advanced data analytics and machine learning dashboard built with **Python** and **Streamlit**.

## **Project Overview**

**CineMetrics** processes a dataset of over 45,000 movies to uncover _hidden trends_, predict <ins>box office revenue</ins>, classify potential hits, and recommend similar movies based on content.

The primary goal of this project is to:
- Provide an **Interactive Dashboard** for high-level metrics (ROI, Budget vs. Revenue).
- Utilize **Machine Learning** to predict financial success.
- Rank movies using a **Weighted Rating Formula**.

---

## **Data Sources**

This project utilizes **The Movies Dataset** from Kaggle.

> "Data is the new oil." — _Clive Humby_

The data was sourced from:
- **Primary Dataset:** [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data)  
  - <sub>Source: Kaggle</sub>  
  - <ins>Description:</ins> Metadata on over 45,000 movies.

---

## **Features**

### **1. 📊 Interactive Dashboard**
- High-level metrics: **Total Movies**, **Avg Budget/Revenue**, <ins>ROI</ins>.
- Financial analysis: Budget vs. Revenue correlations, decade trends, yearly growth.
- Genre & Studio analytics:
  - Genre word clouds.
  - ROI by Genre.
- Distributions: Boxen plots (Popularity), Violin plots (Runtime).
- Data mining: Seasonality (best release month), Popularity vs. Quality.

---

### **2. 🤖 Machine Learning**
- **Success Classifier:**
  - Gradient Boosting model that classifies movies as **Hit** or **Average/Flop**.
  - Includes a <ins>Confusion Matrix</ins>.

---

### **3. 🌟 Top Charts (Ranking Algorithm)**
- Uses the **IMDB Weighted Rating Formula**.
- Balances rating with vote count.
- Genre filtering (e.g., *Top 10 Horror Movies*).

---

# **Setup & Installation**

Follow these instructions to set up the project on your local machine.

---

## **1. Prerequisites**
Ensure you have **Python 3.8+** installed.

---

## **2. Clone the Repository**
Clone the GitHub repository:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <REPOSITORY_FOLDER_NAME>

## 3. Install Dependencies

Run the following command to install all required libraries:

```bash
pip install -r requirements.txt

## 3. Run the App

Launch the dashboard with Streamlit:

```bash
streamlit run app.py

This will open the application at:

**http://localhost:8501**


