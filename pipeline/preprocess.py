import numpy as np 
import pandas as pd

def load_data(sales_path, items_path):
    """Load the sales data"""

    sales = pd.read_csv(sales_path)
    items = pd.read_csv(items_path)
    return sales, items

def clean_data(sales):
    """Clean data sales"""
    # drop data with negative value
    sales = sales[sales['item_cnt_day'] >= 0]

    # cap item_price & item_cnt_day
    sales['item_price'] = np.clip(sales['item_price'], None, sales['item_price'].quantile(0.99))
    sales['item_cnt_day'] = np.clip(sales['item_cnt_day'], None, sales['item_cnt_day'].quantile(0.99))

    return sales

def aggregate_data_monthly(sales):
    """Aggregate daily sales data to monthly sales data"""

    sales['date'] = pd.to_datetime(sales['date'], format="%d.%m.%Y")
    sales['month'] = sales['date'].dt.to_period("M")
    monthly_sales = sales.groupby(['month', 'shop_id', 'item_id']).agg({
        'item_cnt_day': 'sum',
        'item_price': 'mean'
    }).reset_index()
    monthly_sales.rename(columns={'item_cnt_day' : 'item_cnt_month'}, inplace=True)
    
    return monthly_sales

def save_cleaned_data(monthly_sales, output_path):
    """Save cleaned data to new CSV file"""

    monthly_sales.to_csv(output_path, index=False)
    print(f"cleaned data saved to {output_path}")
    
if __name__ == "__main__":
    # path I/O files
    sales_path = "data/sales_train.csv"
    items_path = "data/items.csv"
    output_path = "data/cleaned_sales.csv"

    # run process
    sales, items = load_data(sales_path, items_path)
    sales = clean_data(sales)
    monthly_sales = aggregate_data_monthly(sales)
    save_cleaned_data(monthly_sales, output_path)