
import tensorflow as tf

def get_model ( num_inputs , num_outputs , num_layers , num_neurons ):

    # Input layer
    ph_input = tf.keras.Input( shape =( num_inputs ,) ,name='input_placeholder')
    # Hidden layers
    hidden_layer = tf.keras.layers.Dense ( num_neurons , activation ='tanh')( ph_input )
    for layer in range ( num_layers ):
        hidden_layer = tf.keras.layers.Dense ( num_neurons , activation ='tanh')( hidden_layer )


    # Output layer
    output = tf.keras.layers.Dense ( num_outputs , activation ='linear',name='output_value')( hidden_layer)
    model = tf.keras.Model ( inputs =[ ph_input ], outputs =[ output ])
    # Optimizer
    my_adam = tf.keras.optimizers.Adam()
    # Compilation
    model.compile ( optimizer =my_adam , loss ={ 'output_value': 'mean_squared_error'})
    return model

def mlp(inputs_train, outputs_train, inputs_validation, outputs_validation, layers, fit_kwargs):
    model = get_model(*(layers))

    history = model.fit(inputs_train, outputs_train, 
                        validation_data=(inputs_validation, outputs_validation),
                        **fit_kwargs)

    return model, history