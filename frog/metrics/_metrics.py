import numpy as np

def AE(x, y):
    # Mean Absolute Error 
    return (abs(x-y))

def MAE(x, y):
    # Mean Absolute Error 
    return np.mean(abs(x-y))

def MSE(x, y):
    # Mean Squared Error 
    return np.mean((x-y)**2)

def RMSE(x, y):
    # Root Mean Squared Error 
    return np.sqrt(np.mean((x-y)**2))

def NRMSE(x, y):
    # Normalized Root Mean Squared Error 
    #return np.sqrt(np.mean((x-y)**2))/np.mean(x)
    return np.sqrt(np.mean((x-y)**2))/(np.max(x)-np.min(x))    

def MAPE(x, y):
    # Mean Absolute Percentage Error 
    return np.mean(abs((x-y)/x))

def R2(x, y):
    # Coefficient of Determination
    return 1 - np.sum((x-y)**2)/np.sum((x-np.mean(y))**2)

def RSE(x, y):
    # Relative Squared Error
    return np.sum((x-y)**2)/ np.sum((x- np.mean(x))**2)

def RAE(x, y):
    # Relative Squared Error
    return np.sum((x-y)**2)/ np.sum((x- np.mean(y))**2)