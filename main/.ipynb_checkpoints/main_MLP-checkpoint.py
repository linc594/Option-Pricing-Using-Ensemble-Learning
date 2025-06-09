### main_MLP.py
import pandas as pd
import numpy as np
from model.MLP import train_model, predict, evaluate_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# 加载数据
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# 数据预处理
def preprocess_data(train_df, test_df, use_sliding_window=False, window_size=10):
    if use_sliding_window:
        train_batches = []
        train_labels = []
        
        # 先使用 optID 进行分组
        for _, group in train_df.groupby('optID'):
            group_values = group.iloc[:, 1:-1].values  # 跳过 optID
            group_labels = group.iloc[:, -1].values   # 标签列
            
            for i in range(len(group_values) - window_size + 1):
                train_batches.append(group_values[i:i+window_size].flatten())
                train_labels.append(group_labels[i+window_size-1])
        
        X_train = np.array(train_batches)
        y_train = np.array(train_labels)
    else:
        # 直接从第 1 列开始取，跳过 optID
        X_train = train_df.iloc[:, 1:-1].values  
        y_train = train_df.iloc[:, -1].values

    # 同样的方式处理 X_test
    X_test = test_df.iloc[:, 1:-1].values  
    y_test = test_df.iloc[:, -1].values

    # 归一化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, scaler

# MLP超参数调优
def hyperparameter_tuning(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    best_params = None
    best_mse = float('inf')
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate_init': [0.001, 0.01]
    }
    
    for hl in param_grid['hidden_layer_sizes']:
        for act in param_grid['activation']:
            for sol in param_grid['solver']:
                for lr in param_grid['learning_rate_init']:
                    model = MLPRegressor(hidden_layer_sizes=hl, activation=act, solver=sol, learning_rate_init=lr, max_iter=1000)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_val)
                    mse = mean_squared_error(y_val, predictions)
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {'hidden_layer_sizes': hl, 'activation': act, 'solver': sol, 'learning_rate_init': lr}
    
    return best_params

# 文件路径
train_path = '../data/train.csv'
test_path = '../data/test.csv'

# 加载数据
train_df, test_df = load_data(train_path, test_path)
use_sliding_window = True  # 是否使用滑动窗口
X_train, y_train, X_test, y_test, scaler = preprocess_data(train_df, test_df)

# 是否调参
use_hyperparameter_tuning = True

if use_hyperparameter_tuning:
    best_params = hyperparameter_tuning(X_train, y_train)
    print(f'Best Hyperparameters: {best_params}')
else:
    best_params = {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'learning_rate_init': 0.001}

model = train_model(X_train, y_train, best_params)
predictions = predict(model, X_test)
mse = evaluate_model(y_test, predictions)
print(f'Mean Squared Error: {mse}')

save_model(model, '../model/save/mlp_model.pkl')
