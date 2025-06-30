# sales_analysis

This project is a comprehensive sales forecasting and exploratory analysis pipeline for a restaurant chain. The goal is to analyze historical sales data, uncover trends, and build machine learning models to forecast future item demand.


# 🍽️ Restaurant Sales Forecasting

This project is part of my capstone work and aims to analyze restaurant sales data and forecast future trends using machine learning and deep learning models.

---

## 📁 Project Structure

.
├── data/
│ ├── items.csv
│ ├── sales.csv
│ └── restaurants.csv
│
├── notebooks/
│ └── sales_forecasting_raw.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── eda.py
│ └── model.py
│
├── output/
│ └── [plots, forecasts, etc.]
│
├── main.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


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

    sales.csv — transaction-level item sales

    items.csv — item metadata (name, cost, kcal)

    restaurants.csv — store information

📁 Folder Descriptions

    data/ — Raw input files

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
