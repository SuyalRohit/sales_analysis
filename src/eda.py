import matplotlib.pyplot as plt
import calendar
import os

def save_plot(fig, filename, folder="output"):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), bbox_inches='tight')

def print_basic_metrics(df):
    print(f"Total Revenue: ₹{df['price'].sum():,.2f}")
    print(f"Total Items Sold: {df['item_count'].sum():,.0f}")
    print(f"Average Price: ₹{df['price'].mean():.2f}")

def top_selling_items(df, items_df, top_n=5):
    best_selling = df.groupby('item_id')['item_count'].sum().nlargest(top_n)
    print("Top Selling Items:")
    for item_id, count in best_selling.items():
        name = items_df.loc[items_df['id'] == item_id, 'name'].iloc[0]
        print(f"- {name} (ID: {item_id}) → Sold: {count}")

def plot_daily_sales(df):
    dw_df = df.groupby('date')['item_count'].sum()
    fig, ax = plt.subplots(figsize=(25, 8))
    ax.plot(dw_df)
    ax.set(title="Daily Item Count Over Time", xlabel="Date", ylabel="Items Sold")
    fig.tight_layout()
    save_plot(fig, "daily_sales.png")
    plt.show()
    return fig, ax

def plot_sales_by_day(df):
    days = list(calendar.day_name)
    w_df = df.groupby('day')['item_count'].sum().reindex(days)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(days, w_df.values, color='skyblue')
    ax.set(title='Item Sales by Day of Week', xlabel='Day of the Week', ylabel='Total Items Sold')
    fig.tight_layout()
    save_plot(fig, "sales_by_day.png")
    plt.show()
    return fig, ax

def plot_sales_by_month(df):
    months = list(calendar.month_name)[1:]
    m_df = df.groupby('month')['item_count'].sum().reindex(months)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(months, m_df.values, color='skyblue')
    ax.set(title='Monthly Sales Performance', xlabel='Month', ylabel='Total Items Sold')
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    save_plot(fig, "sales_by_month.png")
    plt.show()
    return fig, ax

def plot_sales_by_quarter(df):
    q_df = df.groupby('quarter')['item_count'].sum()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(q_df.index, q_df.values, color='skyblue')
    ax.set(title='Sales Trends by Quarter', xlabel='Quarter', ylabel='Total Items Sold')
    ax.set_xticks(q_df.index)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    fig.tight_layout()
    save_plot(fig, "sales_by_quarter.png")
    plt.show()
    return fig, ax

def plot_sales_by_quarter_year(df):
    order = sorted(df['quarter_year'].unique())
    y_df = df.groupby('quarter_year')['item_count'].sum().reindex(order).dropna()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(y_df.index, y_df.values, color='skyblue')
    ax.set(title='Sales Trends Over Time', xlabel='Quarter-Year', ylabel='Total Items Sold')
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    save_plot(fig, "sales_by_quarter_year.png")
    plt.show()
    return fig, ax

def plot_sales_per_restaurant(df):
    total_sales = df.groupby('restaurant_name')['price'].sum().sort_values()
    top_restaurant = total_sales.idxmax()
    max_sales = total_sales.max()
    print(f"The top-performing restaurant is {top_restaurant} with sales of ₹{max_sales:,.2f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    total_sales.plot(kind='barh', color='skyblue', ax=ax)
    ax.set(title='Total Sales per Restaurant', xlabel='Total Sales (₹)', ylabel='Restaurant')
    fig.tight_layout()
    save_plot(fig, "sales_per_restaurant.png")
    plt.show()
    return fig, ax

def plot_restaurant_sales_over_time(df):
    month_name = list(calendar.month_name)[1:]

    for name in df['restaurant_name'].unique():
        filtered_df = df[df['restaurant_name'] == name]

        yearly_sales = filtered_df.groupby('year')['price'].sum()
        print(f"\nThe total sales for {name} is ₹{yearly_sales.sum():,.2f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(yearly_sales.index, yearly_sales.values, color='skyblue')
        ax.set(title=f"{name} - Yearly Sales", xlabel="Year", ylabel="Total Sales")
        fig.tight_layout()
        save_plot(fig, f"{name}_yearly_sales.png")
        plt.show()

        monthly_sales = filtered_df.groupby('month')['price'].sum().reindex(month_name)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(monthly_sales.index, monthly_sales.values, color='skyblue')
        ax.set(title=f"{name} - Monthly Sales", xlabel="Month")
        plt.setp(ax.get_xticklabels(), rotation=45)
        fig.tight_layout()
        save_plot(fig, f"{name}_monthly_sales.png")
        plt.show()

        daily_sales = filtered_df.groupby('date')['price'].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_sales.index, daily_sales.values, color='skyblue')
        ax.set(title=f"{name} - Daily Sales", xlabel="Date")
        fig.tight_layout()
        save_plot(fig, f"{name}_daily_sales.png")
        plt.show()

        print("-" * 100)

def analyze_most_popular_items(df, items_df):
    item_counts = df.groupby('item_id')['item_count'].sum()
    most_popular = items_df.loc[items_df['id'] == item_counts.idxmax(), 'name'].iloc[0]

    print(f"\nMost popular item overall: {most_popular}")
    print(f"Sold in stores: {df.loc[df['item_name'] == most_popular, 'restaurant_name'].unique()}")

    for name in df['restaurant_name'].unique():
        filtered_df = df[df['restaurant_name'] == name]
        top_item_id = filtered_df.groupby('item_id')['item_count'].sum().idxmax()
        top_item_name = items_df.loc[items_df['id'] == top_item_id, 'name'].iloc[0]
        print(f"- {name}: {top_item_name}")

def analyze_revenue_vs_sales_volume(df):
    store_revenue = df.groupby('restaurant_name')['price'].sum()
    top_store = store_revenue.idxmax()
    print(f"\nHighest total revenue: {top_store} with ₹{store_revenue.max():,.2f}\n")

    for name in df['restaurant_name'].unique():
        daily_sales = df[df['restaurant_name'] == name].groupby('date')['price'].sum()
        print(f"Highest daily earning of {name}: ₹{daily_sales.max():,.2f}")

def check_unique_prices(items_df):
    if items_df['cost'].nunique() == len(items_df):
        print("All prices are unique.")
    else:
        print("Duplicate prices found.")

def analyze_expensive_items(df, items_df, tolerance=0.01):
    df = df.copy()
    df['cost_per_unit'] = df['price'] / df['item_count']

    for name in df['restaurant_name'].unique():
        max_cost = round(df.loc[df['restaurant_name'] == name, 'cost_per_unit'].max(), 2)
        match = items_df[(items_df['cost'] - max_cost).abs() < tolerance]

        if not match.empty:
            item_name = match['name'].iloc[0]
            kcal = match['kcal'].iloc[0]
            print(f"{name} → Most expensive item: {item_name} ({max_cost} ₹), Calories: {kcal} kcal")
        else:
            print(f"{name} → No matching item found for {max_cost} ₹")
