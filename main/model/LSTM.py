### LSTM.py
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np

# 训练模型
def train_model(X_train, y_train, best_params):
    model = Sequential()
    model.add(LSTM(best_params['units'], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=best_params['optimizer'], loss='mse')
    model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)
    return model

# 预测
def predict(model, X_test):
    return model.predict(X_test)

# 评估
def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    smape = np.mean(2 * np.abs(y_test - predictions) / (np.abs(y_test) + np.abs(predictions))) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'SMAPE': smape
    }

# 保存模型
def save_model(model, mse, rmse, smape, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_data = {
        "model": model,
        "mse": mse,
        "rmse": rmse,
        "smape": smape,
    }
    joblib.dump(model_data, path)

# 加载模型
def load_model(path):
    return tf.keras.models.load_model(path)