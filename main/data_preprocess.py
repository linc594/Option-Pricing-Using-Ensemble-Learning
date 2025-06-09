import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def apply_sliding_window(df, window_size):
    batches = []
    labels = []
    
    for _, group in df.groupby('optID'):
        group_values = group.iloc[:, 1:-1].values  # 跳过 optID
        group_labels = group.iloc[:, -1].values   # 标签列
        
        for i in range(len(group_values) - window_size + 1):
            batches.append(group_values[i:i+window_size].flatten())
            labels.append(group_labels[i+window_size-1])
    
    return np.array(batches), np.array(labels)

def preprocess_data(train_path, test_path, save_dir, use_sliding_window=False, window_size=10):
    train_df, test_df = load_data(train_path, test_path)
    
    if use_sliding_window:
        X_train, y_train = apply_sliding_window(train_df, window_size)
        X_test, y_test = apply_sliding_window(test_df, window_size)
    else:
        X_train = train_df.iloc[:, 1:-1].values  
        y_train = train_df.iloc[:, -1].values
        X_test = test_df.iloc[:, 1:-1].values  
        y_test = test_df.iloc[:, -1].values
    
    # 归一化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 保存处理后的数据
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train_noise.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train_noise.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(save_dir, 'scaler.npy'), scaler.mean_)
    
    print(f"Processed data saved in {save_dir}")
    
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, "../data/train_data_m/data_noise.csv")
    test_path = os.path.join(BASE_DIR, "../data/test_data_m/data.csv")
    save_dir = os.path.join(BASE_DIR, "../processed_data")
    
    preprocess_data(train_path, test_path, save_dir, use_sliding_window=True)
