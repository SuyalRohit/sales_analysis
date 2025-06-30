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

def check_nulls(df_dict):
    for name, df in df_dict.items():
        print(f"\nNull values in {name}:\n{df.isnull().sum()}")

def merge_data(sales, items, restaurants):
    merged_data = pd.merge(sales, items, left_on='item_id', right_on='id', how='left')
    new_df = pd.merge(merged_data, restaurants, left_on='store_id', right_on='id')
    new_df.rename(columns={"name_x": "item_name", "name_y": "restaurant_name"}, inplace=True)
    new_df.drop(columns=['id_x', 'id_y', 'cost'], inplace=True)
    return new_df

def prepare_data():
    """Loads and processes data in a clean pipeline."""
    restaurants, sales, items = load_data()
    sales = preprocess_sales(sales)
    final_df = merge_data(sales, items, restaurants)
    return final_df
