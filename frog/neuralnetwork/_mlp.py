
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
        loss_kwargs: dict=None,        
        random_state=42,
        regularizer=None,
        regularizer_lambda=None,
        dropout=None,
        learning_rate=None,
        fit_kwargs=None):

        self.fit_kwargs = fit_kwargs
    
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
        if isinstance(optimizer, str) and optimizer.upper() == 'adamw_leslie_smith'.upper():
            import tensorflow_addons as tfa
            from one_cycle_tf import OneCycle
            # 1. Set as maximal_learning_rate the values from lr_finder
            # 2. Set as initial_learing_rate = maximal_learning_rate / 25.0 (best practice from fast AI)
            # 3. The size of cycle in iterations (The one cycle will be withing 10 epoch in example below)
            # 4. The shift peak affects the ratio between growing and decaying part of learning rate
            # (in the example below shift_peak=0.3, which means the learning rate will grow to 
            # maximal_learning_rate withing shift_peak * cycle_size = 0.3 * 10 = 3 epoch)
            # 5. final_lr_scale - the scale of value to decay
            # (in case if you want to decay more than initial value or less) 
            # filal_lr = initial_learning_rate * final_lr_scale
            epoch = self.fit_kwargs['regressor__epochs']
            batch_size = self.fit_kwargs['regressor__batch_size']
            dataset_size = self.fit_kwargs['regressor__dataset_size']
            
            number_iterations_per_epoch = dataset_size/batch_size
            cycle_size = epoch * number_iterations_per_epoch
            
            initial_learning_rate = 0.001
            lr_scheduler = OneCycle(initial_learning_rate=initial_learning_rate/25.0,
                                    maximal_learning_rate=initial_learning_rate,
                                    cycle_size=cycle_size, 
                                    shift_peak=initial_learning_rate*10,
                                    final_lr_scale=1.0
                                    )
            # The example of adamw optimizer (adam with decoupled weight decay)
            # for tensorflow2.0 (you need to pre install tensorflow_addons)
            weight_decay=1e-4
            optimizer = tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=weight_decay)

            # This is continuation of the code snippets above.
            max_momentum = 0.95
            min_momentum = 0.85
            momentum_scheduler = OneCycle(initial_learning_rate=max_momentum,
                                    maximal_learning_rate=min_momentum,
                                    cycle_size=cycle_size,
                                    shift_peak=initial_learning_rate*10,
                                    final_lr_scale=1.0
                                    )
            optimizer._set_hyper("beta_1", lambda: momentum_scheduler(optimizer.iterations))
            
            # This is continuation of the code snippets above.
            max_wd = 1e-4
            min_wd = 0.0
            wd_scheduler = OneCycle(initial_learning_rate=max_wd,
                                    maximal_learning_rate=min_wd,
                                    cycle_size=cycle_size, 
                                    shift_peak=initial_learning_rate*10,
                                    final_lr_scale=1.0
                                )
            optimizer._set_hyper("weight_decay", lambda: wd_scheduler(optimizer.iterations))

        # Compilation
        if loss in ['pod_rec_mse']:
            for key, val in loss_kwargs.items():
                if type(val) == str:
                    loss_kwargs[key] = eval(val)
            loss = self.make_pod_reconstruction_loss(loss_type='pod_rec_mse', **loss_kwargs)

        self.model = model
        self.regularizer = regularizer
        self.regularizer_lambda = regularizer_lambda
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.dropout = dropout
        self.loss_kwargs = loss_kwargs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.fit_kwargs = fit_kwargs

        self.compile()

    def compile(self):
        self.model.compile ( optimizer = self.optimizer , loss = self.loss)

        #ray.init(ignore_reinit_error=True)
    def make_pod_reconstruction_loss(self, loss_type, rec_loss_weight=None ):
        """
        Cria uma função de perda baseada na reconstrução POD.

        Args:
            phi: matriz de modos POD (numpy array ou tensor), shape (N, r)
            loss_type: tipo de perda ('mse', 'huber', 'logcosh')
            delta: parâmetro do Huber loss (quando usado)
            alpha: peso para regularização L2 nos coeficientes (y_pred)

        Returns:
            Função de perda customizada para usar no Keras.
        """
        
        def loss(y_true, y_pred):
            import tensorflow as tf
            import numpy as np

            whiten = self.rom.named_steps['reducer'].whiten
            components = self.rom.named_steps['reducer'].components_
            mean = self.rom.named_steps['reducer'].mean_
            if whiten:
                explained_variance = self.rom.named_steps['reducer'].explained_variance_[:, np.newaxis]
                scaled_components = tf.sqrt(explained_variance) * components
                components = scaled_components
                components = tf.cast(components, y_true.dtype)

            reconstruct = lambda X: tf.matmul(X, components ) + mean
            y_true_rec = reconstruct(y_true)
            y_pred_rec = reconstruct(y_pred)

            # Calcula a perda no espaço do campo
            rec_loss = tf.keras.losses.mse(y_true_rec, y_pred_rec)
            pred_loss = tf.keras.losses.mse(y_true, y_pred)
            #rec_loss_weight = 0.5

            if self.regularizer_lambda:
                # Regularização L2 nos coeficientes (penaliza valores grandes)
                reg_loss = tf.reduce_mean(tf.square(y_pred))

                # Loss total
                #return base_loss + alpha * reg_loss
                total_loss =  rec_loss_weight*rec_loss + (1-rec_loss_weight)*pred_loss + self.regularizer_lambda * reg_loss
            else:
                total_loss =  rec_loss_weight*rec_loss + (1-rec_loss_weight)*pred_loss
                
            return total_loss

        return loss

    def fit(self, X, y, **fit_kwargs):
        # Criar diretório único para o experimento
        

        return self.model.fit(X, y,**fit_kwargs)

    def predict(self, X):
        return self.model.predict(X)