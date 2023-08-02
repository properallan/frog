class FR:
    def __init__(self, X_rom, y_rom, surrogate):
        self.X_rom = X_rom
        self.y_rom = y_rom
        self.surrogate = surrogate

    def fit(self, X, y, fit_kwargs : dict = {}):
        print('Performing ROM fit on X data')
        X = self.X_rom.fit_transform(X)
        print('Performing ROM fit on y data')
        y = self.y_rom.fit_transform(y)

        if 'validation_data' in fit_kwargs.keys():
            X_validation = fit_kwargs['validation_data'][0]
            y_validation = fit_kwargs['validation_data'][1]
            print('Performing ROM fit on X validation data')
            X_validation = self.X_rom.transform(X_validation)
            print('Performing ROM fit on y validation data')
            y_validation = self.y_rom.transform(y_validation)
            fit_kwargs['validation_data'] = (X_validation, y_validation)
            
        print('Performing surrogate model fit')
        self.surrogate.fit(
            X, 
            y, 
            **fit_kwargs
        )
    
    def predict(self, X):
        X_in = self.X_rom.transform(X)
        y_out = self.surrogate.predict(X_in)
        y_out = self.y_rom.inverse_transform(y_out)
        
        return y_out