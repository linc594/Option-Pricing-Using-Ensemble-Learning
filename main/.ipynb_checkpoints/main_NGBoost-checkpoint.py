### main_NGBoost.py
import pandas as pd
import numpy as np
from model.NGBoost import train_model, predict, evaluate_model, save_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ngboost import NGBRegressor

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
    
# NGBoost超参数调优
def hyperparameter_tuning(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    best_params = None
    best_mse = float('inf')
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.1, 0.2],
        'natural_gradient': [True, False]
    }
    
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for nat_grad in param_grid['natural_gradient']:
                model = NGBRegressor(n_estimators=n_est, learning_rate=lr, natural_gradient=nat_grad)
                model.fit(X_train, y_train)
                predictions = model.predict(X_val)
                mse = mean_squared_error(y_val, predictions)
                
                if mse < best_mse:
                    best_mse = mse
                    best_params = {'n_estimators': n_est, 'learning_rate': lr, 'natural_gradient': nat_grad}
    
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
    best_params = {'n_estimators': 1000, 'learning_rate': 0.1, 'natural_gradient': True}

model = train_model(X_train, y_train, best_params)
predictions = predict(model, X_test)
mse = evaluate_model(y_test, predictions)
print(f'Mean Squared Error: {mse}')

save_model(model, '../model/save/ngboost_model.pkl')
