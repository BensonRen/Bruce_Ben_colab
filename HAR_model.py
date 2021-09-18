import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
from glob import glob
from tqdm import tqdm_notebook as tqdm
from data_reader import read_data_bruce
from flag_reader import read_flag
from sklearn.metrics import mean_squared_error as mse

def har(Xs, Y):

    cutoff = 1237
    d_lag = 1*390
    w_lag = 5*390
    m_lag = 21*390

    d_x = Xs[:, :d_lag]
    d_x = np.sum(d_x**2, axis = 1).reshape(-1, 1)
    w_x = Xs[:, d_lag:w_lag]
    w_x = np.sum(w_x**2, axis = 1).reshape(-1, 1)
    m_x = Xs[:, w_lag:m_lag]
    m_x = np.sum(m_x**2, axis = 1).reshape(-1, 1)

    X = np.hstack([d_x, w_x, m_x])

    X_cons = sm.add_constant(X)

    #Split in sample and out of sample.
    Y_insp = Y[:cutoff]
    Y_oos = Y[cutoff:]

    X_cons_insp = X_cons[:cutoff]
    X_cons_oos = X_cons[cutoff:]
    print(np.shape(X_cons_insp)) 
    print(X_cons_insp)
    print(np.shape(X_cons_oos)) 
    print(X_cons_oos)
    #quit()
    har_model = sm.OLS(Y_insp, X_cons_insp).fit()

    Y_insp_pred = har_model.predict(X_cons_insp)
    Y_oos_pred = har_model.predict(X_cons_oos)

    insp_mse = mse(Y_insp, Y_insp_pred)
    oos_mse = mse(Y_oos, Y_oos_pred)
    # return Y_insp, Y_insp_pred, Y_oos, Y_oos_pred
    return insp_mse, oos_mse



if __name__ == '__main__':
    # Read flag
    flags = read_flag()
    # read data
    data_x, data_y = read_data_bruce(flags, get_raw_data=True)
    # Run the HAR model on this
    insp_mse, oos_mse = har(data_x, data_y)
    print('insample MSE = {}, outsample MSE = {}'.format(insp_mse, oos_mse))

    
