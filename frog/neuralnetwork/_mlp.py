
import tensorflow as tf
from typing import Union

def get_model ( 
    num_inputs : int, 
    num_outputs : int, 
    num_layers : int, 
    num_neurons : int,
    activation : str = 'tanh',
    optimizer : Union[callable, str] = tf.keras.optimizers.Adam(),
    loss : dict = { 'output_value': 'mean_squared_error'}):

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