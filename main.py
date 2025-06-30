import os
import pandas as pd
from src.preprocessing import prepare_data
from src.eda import (
    print_basic_metrics,
    plot_daily_sales,
    plot_sales_by_day,
    plot_sales_by_month,
    plot_sales_by_quarter,
    plot_sales_by_quarter_year,
    plot_sales_per_restaurant,
    analyze_most_popular_items,
    analyze_revenue_vs_sales_volume,
    check_unique_prices,
    analyze_expensive_items,
)
from src.model import (
    prepare_time_series,
    split_train_test,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    compare_models_plot,
)

# Create output folder
os.makedirs("output", exist_ok=True)

# Step 1: Load and preprocess
df = prepare_data()
df.to_csv("output/final_preprocessed.csv", index=False)

# Step 2: Basic Metrics and EDA
print_basic_metrics(df)
plot_daily_sales(df)
plot_sales_by_day(df)
plot_sales_by_month(df)
plot_sales_by_quarter(df)
plot_sales_by_quarter_year(df)
plot_sales_per_restaurant(df)
analyze_most_popular_items(df, pd.read_csv("data/items.csv"))
analyze_revenue_vs_sales_volume(df)
check_unique_prices(pd.read_csv("data/items.csv"))
analyze_expensive_items(df, pd.read_csv("data/items.csv"))

# Step 3: Prepare time series
ts_df = prepare_time_series(df)
X_train, y_train, X_test, y_test, test_df = split_train_test(ts_df)

# Step 4: Train and Evaluate Models
lr_model = train_linear_regression(X_train, y_train)
rfr_model, _, rfr_scores = train_random_forest(X_train, y_train, X_test, y_test, test_df, ts_df)
xgb_model, _, xgb_scores = train_xgboost(X_train, y_train, X_test, y_test, test_df, ts_df)

# Step 5: Compare Results
results_df = pd.DataFrame({
    "Random Forest": rfr_scores,
    "XGBoost": xgb_scores,
}, index=["RMSE", "MAE", "R2_Score"])
results_df.to_csv("output/model_results.csv")
compare_models_plot(results_df)

print("\nProject run completed. Results saved in /output")
