import pandas as pd
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

    return time_series

def split_train_test(ts_df, cutoff='2021-07-01'):
    """Split time series into train and test sets."""
    train = ts_df[ts_df.index < cutoff]
    test = ts_df[ts_df.index >= cutoff]

    x_cols = ts_df.drop(columns='item_count').columns
    y_col = 'item_count'

    return train[x_cols], train[y_col], test[x_cols], test[y_col], test

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, test_df):
    """Evaluate and visualize model performance."""
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

def forecast_future(model, time_series, x_var, forecast_periods=365):
    last_date = time_series.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_periods + 1)]

    X_forecast = pd.DataFrame(index=future_dates, columns=x_var)
    X_forecast['day'] = X_forecast.index.day
    X_forecast['quarter'] = X_forecast.index.quarter
    X_forecast['year'] = X_forecast.index.year
    X_forecast['month'] = X_forecast.index.month_name()
    X_forecast['day_year'] = X_forecast.index.day_of_year
    X_forecast['day_month'] = X_forecast.index.day
    X_forecast['week_num'] = X_forecast.index.isocalendar().week.astype(int)

    # Encode months
    encoder = OrdinalEncoder(categories=[list(X_forecast['month'].unique())])
    encoder.fit(X_forecast[['month']])
    X_forecast['month'] = encoder.transform(X_forecast[['month']])

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

def build_lstm_model(hp):
    model = Sequential()
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, default=1)

    for i in range(num_layers):
        return_seq = (i != num_layers - 1)
        units = hp.Int(f'units_{i}', min_value=50, max_value=200, step=10)
        if i == 0:
            model.add(LSTM(units=units, return_sequences=return_seq, input_shape=(12, 1)))
        else:
            model.add(LSTM(units=units, return_sequences=return_seq))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def tune_lstm_model(scaled_train):
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=12, batch_size=1)
    tuner = RandomSearch(
        build_lstm_model,
        objective='val_loss',
        max_trials=10,
        directory='my_dir',
        project_name='lstm_tuning'
    )
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    tuner.search(generator, epochs=100, validation_split=0.1, callbacks=[early_stopping])
    return tuner.get_best_hyperparameters()[0]

def train_best_lstm(hp, scaled_data):
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=12, batch_size=1)
    model = build_lstm_model(hp)
    model.fit(generator, epochs=100, callbacks=[EarlyStopping(monitor='loss', patience=2)])
    return model

def predict_lstm(model, scaled_train, scaled_test, test_df, scaler):
    length = 12
    n_features = 1
    test_predictions = []

    current_batch = scaled_train[-length:].reshape((1, length, n_features))
    for i in range(len(scaled_test)):
        current_pred = model.predict(current_batch)[0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)
    test_df = test_df.copy()
    test_df['Predictions'] = true_predictions

    plt.figure(figsize=(10,5))
    plt.plot(test_df.index, test_df['price'], label='Actual')
    plt.plot(test_df.index, test_df['Predictions'], label='Predicted')
    plt.legend()
    plt.title("LSTM Forecast")
    plt.show()

    return test_df

def forecast_with_lstm(model, original_df, synthetic_df):
    scaler = MinMaxScaler()
    scaler.fit(original_df)
    full_scaled = scaler.transform(original_df)

    forecast_scaled = scaler.transform(synthetic_df[:93])
    forecast_generator = TimeseriesGenerator(forecast_scaled, forecast_scaled, length=3, batch_size=1)

    predictions = model.predict(forecast_generator)
    predictions_original = scaler.inverse_transform(predictions)

    forecast_dataset = synthetic_df.iloc[:90].copy()
    forecast_dataset['forecasted_price'] = predictions_original.flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(original_df.index, original_df['price'], label='Actual')
    plt.plot(forecast_dataset.index, forecast_dataset['price'], label='Synthetic', linestyle='--')
    plt.plot(forecast_dataset.index, forecast_dataset['forecasted_price'], label='Forecasted', linestyle='-.')
    plt.title('LSTM Forecast using Synthetic Data')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast_dataset
