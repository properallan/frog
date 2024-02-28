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

def RMSEP(x, y):
    # Root Mean Squared Error Percentage
    eps = np.finfo(np.float64).eps
    return np.sqrt(np.mean((x-y)**2/np.maximum(np.abs(x), eps)))

def NRMSE(x, y):
    # Normalized Root Mean Squared Error 
    #return np.sqrt(np.mean((x-y)**2))/np.mean(x)
    eps = np.finfo(np.float64).eps
    return np.sqrt(MSE(x,y))/np.maximum(np.abs(x), eps).mean()

def MAPE(y_true, y_pred):    
    # Mean Absolute Percentage Error 
    return np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), np.finfo(np.float64).eps))

def R2(x, y):
    # Coefficient of Determination
    eps = np.finfo(np.float64).eps
    num = np.sum((x-y)**2, axis=0)
    den = np.maximum(np.sum((x-np.mean(x, axis=0))**2, axis=0), eps)
    den = np.maximum(den, eps)
    return 1-(num/den).mean()

def RSE(x, y):
    # Relative Squared Error
    return np.sum((x-y)**2)/ np.sum((x- np.mean(x))**2)

def RAE(x, y):
    # Relative Squared Error
    return np.sum((x-y)**2)/ np.sum((x- np.mean(y))**2)


def SMAPE(x, y):
    # Symmetric Mean Absolute Percentage Error 
    eps = np.finfo(np.float64).eps
    return np.mean(np.abs(x-y)/np.maximum((np.abs(x)+np.abs(y))/2), eps)

def MAXPE(y_true, y_pred):
    # Maximum Absolute Percentage Error
    return np.max(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), np.finfo(np.float64).eps))