import pandas as pd

def load_data():
    """
    Loads CSV files for restaurants, sales, and items.
    Returns three DataFrames.
    """
    restaurants = pd.read_csv('data/restaurants.csv')
    sales = pd.read_csv('data/sales.csv')
    items = pd.read_csv('data/items.csv')
    return restaurants, sales, items

def preprocess_sales(sales_df):
    """
    Converts the 'date' column to datetime and adds useful datetime features.
    Handles invalid dates by coercing errors to NaT.
    """
    sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')
    # Optionally drop or handle rows with invalid dates
    sales_df = sales_df.dropna(subset=['date'])
    
    # Add useful datetime columns for analysis
    sales_df['day'] = sales_df['date'].dt.day_name()
    sales_df['month'] = sales_df['date'].dt.month_name()
    sales_df['quarter'] = sales_df['date'].dt.quarter
    sales_df['year'] = sales_df['date'].dt.year
    sales_df['quarter_year'] = sales_df['year'].astype(str) + "-Q" + sales_df['quarter'].astype(str)
    return sales_df

def check_duplicates(df_dict):
    """
    Prints the number of duplicate rows in each DataFrame in df_dict.
    """
    for name, df in df_dict.items():
        print(f"Duplicates in {name}: {df.duplicated().sum()}")

def check_nulls(df_dict):
    """
    Prints the count of null values in each column of each DataFrame in df_dict.
    """
    for name, df in df_dict.items():
        print(f"\nNull values in {name}:\n{df.isnull().sum()}")

def merge_data(sales, items, restaurants):
    """
    Merges sales with items and restaurants DataFrames.
    Drops unnecessary columns and renames for clarity.
    """
    merged_data = pd.merge(sales, items, left_on='item_id', right_on='id', how='left', suffixes=('_sales', '_item'))
    merged_data = pd.merge(merged_data, restaurants, left_on='store_id', right_on='id', how='left', suffixes=('', '_restaurant'))
    
    # Rename for clarity
    merged_data.rename(columns={
        "name_item": "item_name",
        "name": "restaurant_name"
    }, inplace=True)
    
    # Drop duplicate or unnecessary columns
    merged_data.drop(columns=['id_sales', 'id_restaurant', 'cost'], errors='ignore', inplace=True)
    
    return merged_data

def prepare_data():
    """
    Complete data preparation pipeline: load, preprocess, merge.
    Returns the final cleaned DataFrame.
    """
    restaurants, sales, items = load_data()
    sales = preprocess_sales(sales)    
    final_df = merge_data(sales, items, restaurants)
    
    return final_df
