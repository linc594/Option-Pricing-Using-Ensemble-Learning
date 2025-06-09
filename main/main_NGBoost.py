### main_NGBoost.py
import os
import pandas as pd
import numpy as np
from model.NGBoost import train_model, predict, evaluate_model, save_model
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor

# 加载数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "../processed_data")

# 添加时间戳，构造唯一模型保存路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"ngboost_model_{timestamp}.pkl"
save_path = os.path.join(BASE_DIR, "save", model_filename) # 目标路径
log_path = os.path.join(BASE_DIR, "experiment_logs.csv")
# 加载预处理后的数据
X_train = np.load(os.path.join(save_dir, 'X_train_noise.npy'))
y_train = np.load(os.path.join(save_dir, 'y_train_noise.npy'))
X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
y_test = np.load(os.path.join(save_dir, 'y_test.npy'))

# 日志记录函数
def log_experiment(log_path, model_type, params, metrics, model_path):
    param_list = [f"{k}: {v}" for k, v in params.items()]
    param_filled = param_list[:3] + [None] * (3 - len(param_list))

    new_record = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_type,
        'param_1': param_filled[0],
        'param_2': param_filled[1],
        'param_3': param_filled[2],
        'MSE': metrics.get('MSE'),
        'RMSE': metrics.get('RMSE'),
        'SMAPE': metrics.get('SMAPE'),
        'model_path': model_path
    }

    if not os.path.exists(log_path):
        df = pd.DataFrame([new_record])
    else:
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    df.to_csv(log_path, index=False)
    print(f"实验日志已记录: {log_path}")
    
# NGBoost超参数调优
def hyperparameter_tuning(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    best_params = None
    best_mse = float('inf')
    param_grid = {
        'n_estimators': [250, 300],
        'learning_rate': [0.1, 0.15],
        'natural_gradient': [True, False]
    }
    
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for nat_grad in param_grid['natural_gradient']:
                model = NGBRegressor(n_estimators=n_est, learning_rate=lr, natural_gradient=nat_grad)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                metrics = evaluate_model(y_val, predictions)
                mse = metrics['MSE']
                if mse < best_mse:
                    best_mse = mse
                    best_params = {'n_estimators': n_est, 'learning_rate': lr, 'natural_gradient': nat_grad}
    
    return best_params


# 是否调参
use_hyperparameter_tuning = False

if use_hyperparameter_tuning:
    best_params = hyperparameter_tuning(X_train, y_train)
    print(f'Best Hyperparameters: {best_params}')
else:
    best_params = {'n_estimators': 1000, 'learning_rate': 0.05, 'natural_gradient': True}

model = train_model(X_train, y_train, best_params)
predictions = predict(model, X_test)
metrics = evaluate_model(y_test, predictions)
mse = metrics['MSE']
rmse = metrics['RMSE']
smape = metrics['SMAPE']
save_model(model, mse, rmse, smape, save_path)
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Symmetric Mean Absolute Percentage Error: {smape}')
print(f"Trying to save model at: {save_path}")

# 日志记录
log_experiment(log_path, "ngboost", best_params, metrics, save_path)