import pandas as pd
import joblib

def load_test_data(test_path):
    """load & prepare test data"""

    test = pd.read_csv(test_path)
    # Tambahkan fitur yang sama seperti saat training (contoh: 'month')

    test['month'] = 10  # Contoh: set ke bulan oktober
    return test

def load_model(model_path):
    """load model data"""
    return joblib.load(model_path)

def predict(model, test):
    """generate prediction"""
    features = test[['shop_id', 'item_id', 'month']]
    predictions = model.predict(features)
    test['item_cnt_month'] = predictions.clip(0,20)
    
    return test

def save_predictions(test, output_path):
    """save predictions to csv file"""
    submission = test[['ID', 'item_cnt_month']]
    submission.to_csv(output_path, index=False)
    print(f"prediction save to {output_path}")

if __name__ == "__main__":
    # paths
    test_path = "data/test.csv"
    model_path = "pipeline/sales_forecasting_model.joblib"
    output_path = "data/submission.csv"
    
    # run process
    test_data = load_test_data(test_path)
    model = load_model(model_path)
    predictions = predict(model, test_data)
    save_predictions(predictions, output_path)