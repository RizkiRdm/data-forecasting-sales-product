import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def load_data(path):
    """Load cleaned data"""
    
    return pd.read_csv(path)

def prepare_data(data):
    """Prepare data features and target for training"""
    
    # features (shop_id, item_id, month) convert to numeric data type / category data type
    data['month'] = pd.to_datetime(data['month']).dt.month
    
    X = data[['shop_id', 'item_id', 'month']]
    y = data['item_cnt_month']

    return X, y

def train_model(X, y):
    """train data using XGBoost"""
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    
    # evaluate model
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse}")

    return model

def savel_model_data(model, output_path):
    """save trained model to joblib file"""
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # paths
    data_path = "data/cleaned_sales.csv"
    model_path = "pipeline/sales_forecasting_model.joblib"
    
    # run training
    data = load_data(data_path)
    x,y = prepare_data(data)
    model = train_model(x,y)
    save_model = savel_model_data(model, model_path)



