import os
import pandas as pd
import numpy as np
from model.LGBM import *
from datetime import datetime
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 加载数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "../processed_data")

# 添加时间戳，构造唯一模型保存路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"lgbm_model_{timestamp}.pkl"
save_path = os.path.join(BASE_DIR, "save", model_filename)
log_path = os.path.join(BASE_DIR, "experiment_logs.csv")

# 加载预处理后的数据
X_train = np.load(os.path.join(save_dir, 'X_train_noise.npy'))
y_train = np.load(os.path.join(save_dir, 'y_train_noise.npy'))
X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
y_test = np.load(os.path.join(save_dir, 'y_test.npy'))

# 转换为带列名的 DataFrame（用于消除 warning）
feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

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

# LGBM超参数调优
def hyperparameter_tuning(X_train_df, y_train):
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train_df, y_train, test_size=0.2, random_state=42)
    best_params = None
    best_mse = float('inf')
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'num_leaves': [20, 31, 40],
        'max_depth': [-1, 10, 20]
    }

    for lr in param_grid['learning_rate']:
        for ne in param_grid['n_estimators']:
            for nl in param_grid['num_leaves']:
                for md in param_grid['max_depth']:
                    model = lgb.LGBMRegressor(
                        learning_rate=lr,
                        n_estimators=ne,
                        num_leaves=nl,
                        max_depth=md,
                        verbose=-1  # 静默训练信息
                    )
                    model.fit(X_train_sub, y_train_sub)
                    predictions = model.predict(X_val_sub)
                    metrics = evaluate_model(y_val_sub, predictions)
                    mse = metrics['MSE']
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'learning_rate': lr,
                            'n_estimators': ne,
                            'num_leaves': nl,
                            'max_depth': md
                        }
    return best_params

# 是否调参
use_hyperparameter_tuning = False

if use_hyperparameter_tuning:
    best_params = hyperparameter_tuning(X_train_df, y_train)
    print(f'Best Hyperparameters: {best_params}')
else:
    best_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'num_leaves': 40, 'max_depth': -1}

# 模型训练 & 评估
model = train_model(X_train_df, y_train, best_params)
predictions = predict(model, X_test_df)
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
log_experiment(log_path, "lgbm", best_params, metrics, save_path)
