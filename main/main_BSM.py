import os
import pandas as pd
import numpy as np
import torch
from tqdm import trange
from model.functions import *
import matplotlib.pyplot as plt
from datetime import datetime
# from functions import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))# 获取 main_XGBoost.py 所在目录
test_path = os.path.join(BASE_DIR, "../data/test_data_m/data_at_M_new.csv")
test_data_BS = pd.read_csv(test_path)
data_list = []
seq_len = 1 
##for语句，从1到.unique()的数量
##for语句，选取在i下对应.unique()的'optID' 并提前做出滑动窗口处理（每个Index对应一个seq_len为10的窗口）
##开截 （10，9） 9为test_data_BS列数 10为seq_len值
##只取最后一行 得（1，9） 增加维度（1，1，9）
##（1，X，9）
for i in trange(test_data_BS['optID'].unique().shape[0]): 
    for index in range(test_data_BS.loc[(test_data_BS['optID'] == test_data_BS['optID'].unique()[i])].shape[0] - seq_len+1): 
        tmp_data = np.array(test_data_BS.loc[(test_data_BS['optID'] == test_data_BS['optID'].unique()[i])][index:index+seq_len])
        data = torch.tensor(np.array(tmp_data[seq_len-1, :], dtype=np.float64())).unsqueeze(0)
        data_list.append(data)

test_data_BS = torch.cat(data_list, dim=0)

BS_price = torch.zeros(size=(test_data_BS.shape[0], 1))
BSM_price = torch.zeros(size=(test_data_BS.shape[0], 1))
for i in trange(test_data_BS.shape[0]):
    maturity = test_data_BS[i, 2]
    vol = test_data_BS[i, 8]
    strike = test_data_BS[i, 4]
    spot = test_data_BS[i, 5]
    r = test_data_BS[i, 7]
    q = test_data_BS[i, 6]
    Price = 0
    price1 = 0
    if test_data_BS[i, 3] == 0:
        price = call_option_pricer(spot, strike, maturity, r, vol)
        price1 = call_option_pricer_Merton(spot, strike, maturity, r, vol, q)
    if test_data_BS[i, 3] == 1:
        price = put_option_pricer(spot, strike, maturity, r, vol)
        price1 = put_option_pricer_Merton(spot, strike, maturity, r, vol, q)
    BS_price[i, :] = price
    BSM_price[i, :] = price1

BS_price = BS_price.squeeze()
BSM_price = BSM_price.squeeze()
test_price = test_data_BS[:, 9]

def BS_metrics(price, test_price):
    corr, map, mape = metrics(price, test_price)
    corr = torch.mean(corr).float()
    map = torch.mean(map).float()
    mape = torch.mean(mape).float()
    mse = torch.mean(torch.square(test_price - price)).float()
    rmse = torch.sqrt(mse).float()

    print("MSE: {:.5f}\nRMSE: {:0.5f}\nMAP: {:0.5f}\nMAPE: {:0.5f}\nCorrelation: {:0.5f}\n".format(mse, rmse, map, mape, corr))
    ##print("MSE: {:.5f}\nRMSE: {:0.5f}\nMAP: {:0.5f}\nCorrelation: {:0.5f}\n".format(mse, rmse, map, corr))
    #begin_plot = 0
    #plot_len = 240
    #plt.plot(np.arange(plot_len), test_price[begin_plot:begin_plot + plot_len].numpy(), color='blue', label="True Price")
    #plt.plot(np.arange(plot_len), BS_price[begin_plot:begin_plot + plot_len].numpy(), color='red', label="Estimated Price")
    #plt.legend()
    #plt.xlabel('Time')
    #plt.ylabel('Option Price')
    #plt.title("Comparison of Estimated Price and True Price")

    #plt.show()

BS_metrics(BS_price, test_price)
BS_metrics(BSM_price, test_price)
