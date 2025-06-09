### main_GA.py
import os
import pandas as pd
import numpy as np
from model.GA import*
from datetime import datetime
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "../processed_data")

# 添加时间戳，构造唯一模型保存路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"ga_model_{timestamp}.pkl"
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

# 进化算法参数
def hyperparameter_tuning(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    
    toolbox.register("attr_int", np.random.randint, 50, 200)
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_int), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_ga, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    
    pop = toolbox.population(n=10)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)
    
    best_individual = tools.selBest(pop, k=1)[0]
    best_params = {'n_estimators': int(best_individual[0]), 'max_depth': int(best_individual[1])}
    return best_params

# 目标函数
def evaluate_ga(individual, X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(n_estimators=int(individual[0]), max_depth=int(individual[1]))
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    metrics = evaluate_model(y_val, predictions)
    mse = metrics['MSE']
    return mse,

# 是否调参
use_hyperparameter_tuning = False

if use_hyperparameter_tuning:
    best_params = hyperparameter_tuning(X_train, y_train)
    print(f'Best Hyperparameters: {best_params}')
else:
    best_params = {'n_estimators': 100, 'max_depth': 7}

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
log_experiment(log_path, "ga", best_params, metrics, save_path)
