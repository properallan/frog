
from typing import Union
from sklearn.base import BaseEstimator, RegressorMixin, MultiOutputMixin




def get_model ( 
    num_inputs : int, 
    num_outputs : int, 
    num_layers : int, 
    num_neurons : int,
    activation : str = 'tanh',
    optimizer : Union[callable, str] = 'adam',
    loss : dict = { 'output_value': 'mean_squared_error'}):
    import tensorflow as tf

    # Input layer
    ph_input = tf.keras.Input( shape =( num_inputs ,) ,name='input_placeholder')
    # Hidden layers
    hidden_layer = tf.keras.layers.Dense ( num_neurons , activation = activation)( ph_input )
    for layer in range ( num_layers ):
        hidden_layer = tf.keras.layers.Dense ( num_neurons , activation = activation)( hidden_layer )


    # Output layer
    output = tf.keras.layers.Dense ( num_outputs , activation ='linear',name='output_value')( hidden_layer)
    model = tf.keras.Model ( inputs =[ ph_input ], outputs =[ output ])
    # Optimizer
    #my_adam = tf.keras.optimizers.Adam()
    # Compilation
    model.compile ( optimizer = optimizer , loss = loss)


    return model

def mlp(inputs_train, outputs_train, inputs_validation, outputs_validation, layers, fit_kwargs):
    model = get_model(*(layers))

    history = model.fit(inputs_train, outputs_train, 
                        validation_data=(inputs_validation, outputs_validation),
                        **fit_kwargs)

    return model, history


class NeuralNetwork(MultiOutputMixin, RegressorMixin, BaseEstimator):
    def __init__(
        self,
        num_inputs : int, 
        num_outputs : int, 
        num_layers : int, 
        num_neurons : int,
        activation : str = 'tanh',
        optimizer : Union[callable, str] = 'adam',
        loss : dict = { 'output_value': 'mean_squared_error'}, 
        random_state=42,
        regularizer=None,
        regularizer_lambda=None,
        dropout=None,
        learning_rate=None):

        import tensorflow as tf
        import ray
        self.random_state = random_state
        tf.random.set_seed(random_state)
        
        if regularizer is not None:
            if regularizer == 'L1': 
                regularizer = tf.keras.regularizers.L1(regularizer_lambda)
            elif regularizer == 'L2':
                regularizer = tf.keras.regularizers.L2(regularizer_lambda)

        # Input layer
        ph_input = tf.keras.Input( shape =( num_inputs ,) ,name='input_placeholder')
        # Hidden layers
        hidden_layer = tf.keras.layers.Dense ( num_neurons , activation = activation, kernel_regularizer = regularizer)( ph_input )
        if dropout: hidden_layer = tf.keras.layers.Dropout(dropout)(hidden_layer)
        for layer in range ( num_layers ):
            hidden_layer = tf.keras.layers.Dense ( num_neurons , activation = activation, kernel_regularizer = regularizer )( hidden_layer )
            if dropout: hidden_layer = tf.keras.layers.Dropout(dropout)(hidden_layer)


        # Output layer
        output = tf.keras.layers.Dense ( num_outputs , activation ='linear',name='output_value')( hidden_layer)
        model = tf.keras.Model ( inputs =[ ph_input ], outputs =[ output ])
        # Optimizer
        #my_adam = tf.keras.optimizers.Adam()
        if isinstance(optimizer, str):
            if optimizer.upper() == 'ADAM' and learning_rate:
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # Compilation
        model.compile ( optimizer = optimizer , loss = loss)

        

        self.model = model

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss

        #ray.init(ignore_reinit_error=True)

    def fit(self, X, y, **fit_kwargs):
        # Criar diretório único para o experimento
        

        return self.model.fit(X, y,**fit_kwargs)

    def predict(self, X):
        return self.model.predict(X)