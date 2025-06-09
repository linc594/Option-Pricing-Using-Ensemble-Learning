import os
import joblib

# 加载保存的模型和结果
def load_model_with_results(model_path):
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    mse = model_data['mse']
    rmse = model_data['rmse']
    return model, mse, rmse

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "save", "xgboost_model.pkl")  # 使用兼容路径拼接方式

    # 加载模型和结果
    model, mse, rmse = load_model_with_results(model_path)

    print(f"Model loaded successfully from {model_path}")
    print(f"Mean Squared Error: {mse}")
    print(f'Root Mean Squared Error: {rmse}')
