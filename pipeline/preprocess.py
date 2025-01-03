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

def aggregate_monthly(sales):
    """"""