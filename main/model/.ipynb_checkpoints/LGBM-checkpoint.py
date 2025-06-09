### LGBM.py
import lightgbm as lgb
import joblib

# 训练模型
def train_model(X_train, y_train, best_params):
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    return model

# 预测
def predict(model, X_test):
    return model.predict(X_test)

# 评估
def evaluate_model(y_test, predictions):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_test, predictions)

# 保存模型
def save_model(model, path):
    joblib.dump(model, path)

# 加载模型
def load_model(path):
    return joblib.load(path)
