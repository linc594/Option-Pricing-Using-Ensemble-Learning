# load_model.py
import joblib

# 加载保存的模型和结果
def load_model_with_results(model_path):
    # 加载保存的模型和结果
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    mse = model_data['mse']
    y_test = model_data['y_test']
    
    return model, mse, y_test

# 如果你希望在执行该脚本时直接加载模型并显示结果
if __name__ == "__main__":
    model_path = '../model/save/catboost_model_with_results.pkl'  # 修改为你的模型路径
    
    # 加载模型和结果
    model, predictions, mse, y_test = load_model_with_results(model_path)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Mean Squared Error: {mse}")
