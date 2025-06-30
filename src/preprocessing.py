import pandas as pd

def load_data():
    restaurants = pd.read_csv('data/restaurants.csv')
    sales = pd.read_csv('data/sales.csv')
    items = pd.read_csv('data/items.csv')
    return restaurants, sales, items

def preprocess_sales(sales_df):
    sales_df['date'] = pd.to_datetime(sales_df['date'], format='mixed')
    return sales_df

def check_duplicates(df_dict):
    for name, df in df_dict.items():
        print(f"Duplicates in {name}: {df.duplicated().sum()}")
