import numpy as np

def AE(y_true, y_pred):
    """
    Absolute Error (AE)
    
    Retorna o erro absoluto ponto a ponto entre y_true e y_pred.
    
    Parâmetros:
        y_true: array-like - valores verdadeiros
        y_pred: array-like - valores preditos
        
    Retorno:
        Erro absoluto (element-wise)
    """
    return abs(y_true - y_pred)


def MAE(y_true, y_pred):
    """
    Mean Absolute Error (MAE)
    
    Média do erro absoluto entre os valores verdadeiros e preditos.
    
    Retorno:
        Escalar com a média do erro absoluto
    """
    return np.mean(abs(y_true - y_pred))


def MSE(y_true, y_pred):
    """
    Mean Squared Error (MSE)
    
    Média dos quadrados dos erros (diferenças ao quadrado).
    
    Retorno:
        Escalar com o erro quadrático médio
    """
    return np.mean((y_true - y_pred) ** 2)


def RMSE(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE)
    
    Raiz quadrada da média dos erros quadráticos.
    
    Retorno:
        Escalar com o RMSE
    """
    return np.sqrt(MSE(y_true, y_pred))


def RMSEP(y_true, y_pred):
    """
    Root Mean Squared Error Percentage (RMSEP)
    
    RMSE normalizado ponto a ponto em relação ao valor absoluto de y_true,
    fornecendo um erro percentual ponto a ponto.
    
    Retorno:
        Escalar com erro percentual quadrático médio
    """
    eps = np.finfo(np.float64).eps
    return np.sqrt(np.mean(((y_true - y_pred) ** 2) / np.maximum(np.abs(y_true), eps)))


def NRMSE(y_true, y_pred):
    """
    Normalized RMSE (NRMSE)
    
    RMSE normalizado pela média do valor absoluto de y_true.
    
    Retorno:
        Escalar com erro quadrático normalizado
    """
    eps = np.finfo(np.float64).eps
    return RMSE(y_true, y_pred) / np.maximum(np.abs(y_true), eps).mean()


def MAPE(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE)
    
    Erro absoluto percentual médio.
    
    Retorno:
        Escalar com o erro percentual médio
    """
    eps = np.finfo(np.float64).eps
    return np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps))*100


def R2(y_true, y_pred):
    """
    Coefficient of Determination (R²)
    
    Mede a proporção da variância explicada pelo modelo.
    
    Retorno:
        Escalar com o valor de R²
    """
    eps = np.finfo(np.float64).eps
    num = np.sum((y_true - y_pred) ** 2, axis=0)
    den = np.maximum(np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0), eps)
    return 1 - (num / den).mean()


def RSE(y_true, y_pred):
    """
    Relative Squared Error (RSE)
    
    Soma dos erros ao quadrado em relação à variância dos dados.
    
    Retorno:
        Escalar com o erro relativo quadrático
    """
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def RAE(y_true, y_pred):
    """
    Relative Absolute Error (RAE)
    
    Soma dos erros absolutos em relação à soma dos desvios absolutos da média dos preditos.
    
    Retorno:
        Escalar com erro absoluto relativo
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_pred)))


def SMAPE(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Métrica percentual que normaliza o erro absoluto pela média dos valores absolutos predito e verdadeiro.
    
    Retorno:
        Escalar com erro percentual absoluto simétrico
    """
    eps = np.finfo(np.float64).eps
    return np.mean(np.abs(y_pred - y_true) / np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, eps))


def MAXPE(y_true, y_pred):
    """
    Maximum Absolute Percentage Error (MAXPE)
    
    Erro percentual absoluto máximo entre os valores.
    
    Retorno:
        Escalar com o maior erro percentual encontrado
    """
    eps = np.finfo(np.float64).eps
    return np.max(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), eps))*100
