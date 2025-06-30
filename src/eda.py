import matplotlib.pyplot as plt
import calendar

def print_basic_metrics(df):
    print(f"Total Revenue: ₹{df['price'].sum():,.2f}")
    print(f"Total Items Sold: {df['item_count'].sum():,.0f}")
    print(f"Average Price: ₹{df['price'].mean():.2f}")

def top_selling_items(df, items_df, top_n=5):
    best_selling = df.groupby('item_id')['item_count'].sum().nlargest(top_n)
    print("Top Selling Items:")
    for item_id, count in best_selling.items():
        name = items_df[items_df['id'] == item_id]['name'].iloc[0]
        print(f"- {name} (ID: {item_id}) → Sold: {count}")

def plot_daily_sales(df):
    dw_df = df.groupby('date')['item_count'].sum()
    plt.figure(figsize=(25, 8))
    plt.plot(dw_df)
    plt.title("Daily Item Count Over Time")
    plt.xlabel("Date")
    plt.ylabel("Items Sold")
    plt.tight_layout()
    plt.show()

def plot_sales_by_day(df):
    df['day'] = df['date'].dt.day_name()
    days = list(calendar.day_name)
    w_df = df.groupby('day')['item_count'].sum().loc[days]
    
    plt.figure(figsize=(8, 4))
    plt.bar(days, w_df.values, color='skyblue')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Items Sold')
    plt.title('Item Sales by Day of Week')
    plt.tight_layout()
    plt.show()

def plot_sales_by_month(df):
    df['month'] = df['date'].dt.month_name()
    months = list(calendar.month_name)[1:]
    m_df = df.groupby('month')['item_count'].sum().loc[months]
    
    plt.figure(figsize=(8, 4))
    plt.bar(months, m_df.values, color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Total Items Sold')
    plt.title('Monthly Sales Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sales_by_quarter(df):
    df['quarter'] = df['date'].dt.quarter
    q_df = df.groupby('quarter')['item_count'].sum()
    
    plt.figure(figsize=(8, 4))
    plt.bar(q_df.index, q_df.values, color='skyblue')
    plt.xlabel('Quarter')
    plt.ylabel('Total Items Sold')
    plt.title('Sales Trends by Quarter')
    plt.xticks(q_df.index, ['Q1', 'Q2', 'Q3', 'Q4'])
    plt.tight_layout()
    plt.show()

def plot_sales_by_quarter_year(df):
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['quarter-year'] = "Q" + df['quarter'].astype(str) + "-" + df['year'].astype(str)
    order = ['Q{}-{}'.format(q, y) for y in sorted(df['year'].unique()) for q in range(1, 5)]
    y_df = df.groupby('quarter-year')['item_count'].sum().reindex(order).dropna()
    
    plt.figure(figsize=(10, 5))
    plt.bar(y_df.index, y_df.values, color='skyblue')
    plt.xlabel('Quarter-Year')
    plt.ylabel('Total Items Sold')
    plt.title('Sales Trends Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sales_per_restaurant(df):
    total_sales = df.groupby('restaurant_name')['price'].sum().sort_values()
    top_restaurant = total_sales.idxmax()
    max_sales = total_sales.max()
    print(f"The top-performing restaurant is {top_restaurant} with sales of ₹{max_sales:,.2f}")

    plt.figure(figsize=(12, 6))
    total_sales.plot(kind='barh', color='skyblue')
    plt.xlabel('Total Sales (₹)')
    plt.ylabel('Restaurant')
    plt.title('Total Sales per Restaurant')
    plt.tight_layout()
    plt.show()
def plot_restaurant_sales_over_time(df):
    month_name = list(calendar.month_name)[1:]  # Ensures correct order

    for name in df['restaurant_name'].unique():
        filtered_df = df[df['restaurant_name'] == name]

        # Yearly Sales
        yearly_sales = filtered_df.groupby('year')['price'].sum()
        print(f"\nThe total sales for {name} is ₹{yearly_sales.sum():,.2f}")

        plt.figure(figsize=(10, 5))
        plt.bar(yearly_sales.index, yearly_sales.values, color='skyblue')
        plt.title(f"{name} - Yearly Sales")
        plt.xlabel("Year")
        plt.ylabel("Total Sales")
        plt.tight_layout()
        plt.show()

        # Monthly Sales
        monthly_sales = filtered_df.groupby('month')['price'].sum()
        monthly_sales = monthly_sales.loc[month_name].squeeze()

        plt.figure(figsize=(10, 5))
        plt.bar(monthly_sales.index, monthly_sales.values, color='skyblue')
        plt.title(f"{name} - Monthly Sales")
        plt.xlabel("Month")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Daily Sales
        daily_sales = filtered_df.groupby('date')['price'].sum()
        plt.figure(figsize=(10, 5))
        plt.plot(daily_sales.index, daily_sales.values, color='skyblue')
        plt.title(f"{name} - Daily Sales")
        plt.xlabel("Date")
        plt.tight_layout()
        plt.show()

        print("-" * 100)

def analyze_most_popular_items(df, items_df):
    item_counts = df.groupby('item_id')['item_count'].sum()
    most_popular = items_df[items_df['id'] == item_counts.idxmax()]['name'].iloc[0]

    print(f"\n🌟 Most popular item overall: {most_popular}")
    print(f"Sold in stores: {df[df['item_name'] == most_popular]['restaurant_name'].unique()}")

    for name in df['restaurant_name'].unique():
        filtered_df = df[df['restaurant_name'] == name]
        top_item_id = filtered_df.groupby('item_id')['item_count'].sum().idxmax()
        top_item_name = items_df[items_df['id'] == top_item_id]['name'].iloc[0]
        print(f"- {name}: {top_item_name}")

def analyze_revenue_vs_sales_volume(df):
    store_revenue = df.groupby('restaurant_name')['price'].sum()
    top_store = store_revenue.idxmax()
    print(f"\n🏆 Highest total revenue: {top_store} with ₹{store_revenue.max():,.2f}\n")

    for name in df['restaurant_name'].unique():
        daily_sales = df[df['restaurant_name'] == name].groupby('date')['price'].sum()
        print(f"💰 Highest daily earning of {name}: ₹{daily_sales.max():,.2f}")

def check_unique_prices(items_df):
    if items_df['cost'].nunique() == len(items_df):
        print("✅ All prices are unique.")
    else:
        print("⚠️ Duplicate prices found.")

def analyze_expensive_items(df, items_df):
    df['cost_per_unit'] = df['price'] / df['item_count']

    for name in df['restaurant_name'].unique():
        max_cost = round(df[df['restaurant_name'] == name]['cost_per_unit'].max(), 2)
        match = items_df[items_df['cost'] == max_cost]

        if not match.empty:
            item_name = match['name'].iloc[0]
            kcal = match['kcal'].iloc[0]
            print(f"🍽️ {name} → Most expensive item: {item_name} ({max_cost} ₹), Calories: {kcal} kcal")
        else:
            print(f"🍽️ {name} → No matching item found for {max_cost} ₹")
