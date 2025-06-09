### main_GA.py
import pandas as pd
import numpy as np
from model.GA import train_model, predict, evaluate_model, save_model
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
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
    mse = mean_squared_error(y_val, predictions)
    return mse,

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
    best_params = {'n_estimators': 100, 'max_depth': 10}

model = train_model(X_train, y_train, best_params)
predictions = predict(model, X_test)
mse = evaluate_model(y_test, predictions)
print(f'Mean Squared Error: {mse}')

save_model(model, '../model/save/ga_model.pkl')
