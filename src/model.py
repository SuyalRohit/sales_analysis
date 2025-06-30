import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from datetime import timedelta
import calendar

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch

month_name = list(calendar.month_name)[1:]
day_name = list(calendar.day_name)

def prepare_time_series(df):
    """Aggregate data and engineer time features for modeling."""
    time_series = df.groupby('date').agg({
        'item_count': 'sum',
        'day': lambda x: x.unique()[0],
        'month': lambda x: x.unique()[0],
        'year': lambda x: x.unique()[0],
        'quarter': lambda x: x.unique()[0]
    })

    time_series['day_year'] = time_series.index.day_of_year
    time_series['day_month'] = time_series.index.day
    time_series['week_num'] = time_series.index.isocalendar().week.astype(int)

    encoder = OrdinalEncoder(categories=[month_name, day_name])
    encoder.fit(time_series[['month', 'day']])
    time_series[['month', 'day']] = encoder.transform(time_series[['month', 'day']])

    return time_series, encoder

def split_train_test(ts_df, cutoff='2021-07-01'):
    train = ts_df[ts_df.index < cutoff]
    test = ts_df[ts_df.index >= cutoff]

    x_cols = ts_df.drop(columns='item_count').columns
    y_col = 'item_count'

    return train[x_cols], train[y_col], test[x_cols], test[y_col], test

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, test_df):
    test_df = test_df.copy()
    test_df['lr_pred'] = model.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.plot(test_df.index, test_df['item_count'], label="Actual")
    plt.plot(test_df.index, test_df['lr_pred'], label="Linear Regression Prediction")
    plt.legend()
    plt.title("Linear Regression Forecast")
    plt.tight_layout()
    plt.show()

    rmse = mean_squared_error(y_test, test_df['lr_pred'], squared=False)
    mae = mean_absolute_error(y_test, test_df['lr_pred'])
    r2 = r2_score(y_test, test_df['lr_pred'])

    print(f"Linear Regression:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2 Score: {r2*100:.2f}")
    return pd.DataFrame([rmse, mae, r2*100], index=['RMSE', 'MAE', 'R2_Score'], columns=['Linear Regression']).round(2)

def evaluate_predictions(y_true, y_pred, label="Model"):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label}:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR2 Score: {r2*100:.2f}")
    return rmse, mae, r2*100

def train_random_forest(X_train, y_train, X_test, y_test, test_df, time_series):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df['rfr_pred'] = model.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.plot(time_series.item_count, label="Actual")
    plt.plot(test_df['rfr_pred'], label="Random Forest Prediction")
    plt.legend()
    plt.title("Random Forest Forecast")
    plt.tight_layout()
    plt.show()

    rmse, mae, r2 = evaluate_predictions(test_df['item_count'], test_df['rfr_pred'], "Random Forest")
    return model, test_df, [rmse, mae, r2]

def train_xgboost(X_train, y_train, X_test, y_test, test_df, time_series):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    test_df = test_df.copy()
    test_df['xgbr_pred'] = model.predict(X_test)

    plt.figure(figsize=(10, 5))
    plt.plot(time_series.item_count, label="Actual")
    plt.plot(test_df['xgbr_pred'], label="XGBoost Prediction")
    plt.legend()
    plt.title("XGBoost Forecast")
    plt.tight_layout()
    plt.show()

    rmse, mae, r2 = evaluate_predictions(test_df['item_count'], test_df['xgbr_pred'], "XGBoost")
    return model, test_df, [rmse, mae, r2]

def compare_models_plot(results_df):
    plt.figure(figsize=(10, 5))
    for col in results_df.columns:
        plt.plot(results_df[col], label=col)
    plt.legend(loc="upper right")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()

def forecast_future(model, time_series, x_var, encoder, forecast_periods=365):
    last_date = time_series.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_periods + 1)]

    X_forecast = pd.DataFrame(index=future_dates, columns=x_var)
    X_forecast['day'] = future_dates_day = X_forecast.index.day_name()
    X_forecast['month'] = future_dates_month = X_forecast.index.month_name()
    X_forecast['quarter'] = X_forecast.index.quarter
    X_forecast['year'] = X_forecast.index.year
    X_forecast['day_year'] = X_forecast.index.day_of_year
    X_forecast['day_month'] = X_forecast.index.day
    X_forecast['week_num'] = X_forecast.index.isocalendar().week.astype(int)

    X_forecast[['month', 'day']] = encoder.transform(X_forecast[['month', 'day']])

    forecast_values = model.predict(X_forecast)
    forecast_df = pd.DataFrame(index=future_dates, data=forecast_values, columns=['forecast'])

    plt.figure(figsize=(25, 5))
    plt.plot(time_series.item_count, label="Actual")
    plt.plot(forecast_df.forecast, label="Future Forecast")
    plt.title('Future Forecast (Next 365 Days)')
    plt.xlabel('Date')
    plt.ylabel('Item Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return forecast_df
