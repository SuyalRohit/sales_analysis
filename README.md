# 🍽️ Restaurant Sales Forecasting

This project is part of my capstone work and aims to analyze restaurant sales data and forecast future trends using machine learning and deep learning models.

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
🚀 How to Run

Run the project end-to-end using:

python main.py --eda --train

Flag	Description
--eda	Runs exploratory data analysis and saves plots to output/
--train	Trains machine learning models and forecasts future sales
--all	Runs both EDA and training pipeline
📊 Exploratory Analysis Highlights

    Total Revenue: ₹X

    Top-Selling Items: Visualized with bar plots

    Sales Trends: Day of week, month, quarter, and restaurant-wise trends

    All plots are saved to the output/ folder.

🤖 Models Implemented
Model	RMSE	MAE	R² Score
Linear Regression			
Random Forest			
XGBoost			
LSTM (Keras + Tuner)			

Forecasts are visualized and saved. LSTM was tuned using keras-tuner for optimal performance.
## 🧪 How to Run

> Run the project using `main.py` by specifying which step you want to execute:

```bash
python main.py --eda         # For visualizations and metrics
python main.py --train       # For training and evaluating models
python main.py --forecast    # For future forecasting

🖼️ Sample Output

Here’s an example of daily sales trend:

🔧 Requirements

Install dependencies with:

pip install -r requirements.txt

📊 Data Sources

    sales.csv — sales info (item_id, item_cost, price)

    items.csv — item metadata (name, cost, kcal)

    restaurants.csv — store information

📁 Folder Descriptions

    data/ — datasets

    src/ — Modular Python scripts for:

        preprocessing.py — data loading, cleaning, merging

        eda.py — metrics, plots, and business insights

        model.py — model training, evaluation, and forecasting

    output/ — saved plots and forecasts for use in the report/README

📌 Notes

    Models can be extended with additional time features.

    CLI argument parsing allows flexibility in running pipeline steps.

    Code is modular and maintainable — ready for scaling or deployment.

📜 License

This project is licensed under the MIT License
