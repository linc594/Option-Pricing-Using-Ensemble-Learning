### main_LSTM.py
import os
import pandas as pd
import numpy as np
from model.LSTM import*
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 加载数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "../processed_data")

# 添加时间戳，构造唯一模型保存路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"lstm_model_{timestamp}.h5"
save_path = os.path.join(BASE_DIR, "save", model_filename) # 目标路径
log_path = os.path.join(BASE_DIR, "experiment_logs.csv")

# 加载预处理后的数据
X_train = np.load(os.path.join(save_dir, 'X_train_noise.npy'))
y_train = np.load(os.path.join(save_dir, 'y_train_noise.npy'))
X_test = np.load(os.path.join(save_dir, 'X_test.npy'))
y_test = np.load(os.path.join(save_dir, 'y_test.npy'))

# 构造 LSTM 所需格式的数据（使用 TimeseriesGenerator）
sequence_length = 10  # 你可以调整这个值
batch_size = 1

generator = TimeseriesGenerator(X_train, y_train, length=sequence_length, batch_size=batch_size)
X_train_seq = np.array([x[0] for x, _ in generator])
y_train_seq = np.array([y[0] for _, y in generator])

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
    
# LSTM超参数调优
def hyperparameter_tuning(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    best_params = None
    best_mse = float('inf')
    param_grid = {
        'units': [50, 75],
        'optimizer': ['adam'],
        'epochs': [50, 100],
        'batch_size': [16, 32]
    }
    
    for units in param_grid['units']:
        for opt in param_grid['optimizer']:
            for ep in param_grid['epochs']:
                for batch in param_grid['batch_size']:
                    model = Sequential([
                    Input(shape=(X_train.shape[1], X_train.shape[2])),
                    LSTM(units, activation='relu'),
                    Dense(1)
                    ])
                    model.compile(optimizer=opt, loss='mse')
                    model.fit(X_train, y_train, epochs=ep, batch_size=batch, verbose=0)
                    predictions = model.predict(X_val)
                    metrics = evaluate_model(y_val, predictions)
                    mse = metrics['MSE']                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'units': units, 'optimizer': opt, 'epochs': ep, 'batch_size': batch}
    
    return best_params

# 是否调参
use_hyperparameter_tuning = False

if use_hyperparameter_tuning:
    best_params = hyperparameter_tuning(X_train_seq, y_train_seq)
    print(f'Best Hyperparameters: {best_params}')
else:
    best_params = {'units': 50, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32}

# 模型训练
# 模型训练
model = train_model(X_train_seq, y_train_seq, best_params)

# 构造测试集序列
test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=1)
X_test_seq = np.array([x[0] for x, _ in test_generator])
y_test_seq = np.array([y[0] for _, y in test_generator])

# 模型预测与评估
predictions = predict(model, X_test_seq)
metrics = evaluate_model(y_test_seq, predictions)
mse = metrics['MSE']
rmse = metrics['RMSE']
smape = metrics['SMAPE']
save_model(model, mse, rmse, smape, save_path)

# 输出结果
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Symmetric Mean Absolute Percentage Error: {smape}')
print(f"Trying to save model at: {save_path}")

# 日志记录
log_experiment(log_path, "lstm", best_params, metrics, save_path)
