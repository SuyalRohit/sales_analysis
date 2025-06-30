# 🍽️ Restaurant Sales Forecasting

This project is part of my capstone work and aims to analyze restaurant sales data and forecast future trends using machine learning and deep learning models.

---
## Install
    pip install pandas==2.2.2, scikit-learn==1.6.1, xgboost==2.1.4, tensorflow==2.18.0, keras-tuner==1.4.7, matplotlib==3.10.0, missingno==0.5.2, openpyxl==3.1.5. xlrd==2.0.2

---

## 🚀 Features

- Data loading and cleaning from multiple CSVs
- Feature engineering and merging of sales, items, and restaurant metadata
- Visual analysis of sales by:
  - Day of week
  - Month
  - Quarter and year
- Forecasting using:
  - Linear Regression
  - Random Forest
  - XGBoost
  - LSTM (with KerasTuner for hyperparameter tuning)

---

## 🧪 How to Run

Run the project using `main.py` by specifying which step you want to execute:

    python main.py --eda         # For visualizations and metrics
    python main.py --train       # For training and evaluating models
    python main.py --forecast    # For future forecasting

🖼️ Sample Output

Here’s an example of daily sales trend:

---

📊 Data Sources

    sales.csv — sales info (item_id, item_cost, price)

    items.csv — item metadata (name, cost, kcal)

    restaurants.csv — store information

---

📁 Folder Descriptions

    data/ — datasets

    src/ — Modular Python scripts for:

        preprocessing.py — data loading, cleaning, merging

        eda.py — metrics, plots, and business insights

        model.py — model training, evaluation, and forecasting

    output/ — saved plots and forecasts for use in the report/README

---

📜 License

This project is licensed under the MIT License
