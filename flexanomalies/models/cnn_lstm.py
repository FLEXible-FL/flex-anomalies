from flexanomalies.models import BaseModel
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import numpy as np
from flexanomalies.utils.metrics import *
class DeepCNN_LSTM(BaseModel):
    """ 
    filters_cnn: list, containing the dimensionality of each CNN 1D layer

    units_lstm: list, containing the dimensionality of each LSTM layer

    kernel_size: list, containing the dimensionality of kernel of each LSTM layer

    hidden_act : str or list, optional default='relu'
         All hidden layers do not necessarily have to use the same activation type. 
         str: the same hidden_act for all layers
         list, size 2: two activation functions, one for all CNN layers and one for all LSTM layers
         list, one activation function for each layer (CNN +LSTM)

    output_act: str, optional (default='linear'),output activation of the final layer


    loss : str or obj, optional (default= 'mse')
        Name of objective function or objective function.

    input_dim: int, number of features


    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.


    epochs : int, optional (default=1)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    validation_size : float, optional (default=0.2)
        The percentage of data to be used for validation.

    callbacks: list, tensorflow callbacks

     contamination : float in (0., 0.5), optional (default=0.1)
         Contamination of the data set, the proportion of outliers in the data set.


    """
    def __init__(
        self,
        input_dim,
        filters_cnn,
        units_lstm,
        kernel_size,
        callbacks=[],
        hidden_act="relu",
        output_act="linear",
        loss="mse",
        validation_size=0.2,
        batch_size=32,
        epochs=1,
        optimizer="adam",
        w_size = 30,
        n_pred = 1,
        model_path = "",
        contamination = 0.1
        
        ) -> None:
        super(DeepCNN_LSTM,self).__init__(contamination = contamination)
        self.input_dim = input_dim
        self.filters_cnn = filters_cnn
        self.units_lstm = units_lstm
        self.kernel_size = kernel_size
        self.w_size = w_size
        self.n_pred = n_pred
        self.callbacks = []
        self.model_path = model_path
        self.update_callbacks(callbacks=callbacks)
        
        # if hidden_act is str 
        self.hidden_act = (
            [hidden_act] * (len(self.filters_cnn)+ len(self.units_lstm))
            if type(hidden_act) is not list
            else hidden_act
        )
        
        # if hidden_act is list and len(list) is 2
        self.hidden_act = (
            ([self.hidden_act[0]] * len(self.filters_cnn)) + ([self.hidden_act[1]]* len(self.units_lstm))
            if len(self.hidden_act)== 2
            else self.hidden_act
        )
        
         # Validate and complete the number of hidden neurons
        self.hidden_act = (
            self.hidden_act
            + [self.hidden_act[-1]] * ((len(self.filters_cnn)+ len(self.units_lstm)) - len(self.hidden_act))
            if (len(self.hidden_act) < (len(self.filters_cnn)+ len(self.units_lstm))) and len(self.hidden_act)!=2
            else self.hidden_act
        )

        self.output_act = output_act
        self.loss = loss
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.model = self._build_model()
    
    def update_callbacks(self, callbacks):
        model_checkpoint_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.005
        )
        self.callbacks = [model_checkpoint_callback] + callbacks


    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(
                layers.Conv1D(
                filters = self.filters_cnn[0],
                kernel_size= self.kernel_size[0],
                activation=self.hidden_act[0],
                input_shape=(self.w_size,self.input_dim)
                )
        )
        model.add(layers.MaxPooling1D(pool_size=2))
        
        # Layers CNN
        for f,kernel, hidden_act in zip(self.filters_cnn[1:],self.kernel_size[1:], self.hidden_act[1:]):
                print(hidden_act,f,kernel)
                model.add(layers.Conv1D(filters = f, kernel_size= kernel, activation= hidden_act))        
                model.add(layers.MaxPooling1D(pool_size=2))
        
        
        # Layers LSTM           
        for u, hidden_act in zip(self.units_lstm, self.hidden_act[len(self.hidden_act)-len(self.units_lstm):]):
                 model.add(layers.LSTM(u, activation= hidden_act, return_sequences=True))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(self.input_dim*self.n_pred, activation= self.output_act))
        model.add(layers.Reshape((self.n_pred,self.input_dim)))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        print(model.summary())
        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods

        """

        #np.random.shuffle(Xscaler)
        self.history = self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_size,
            verbose=1,
        ).history


    
    def predict(self, X,y):
       
        prediction = self.model.predict(X)
         
        self.d_scores_ = (np.linalg.norm(y - prediction, axis = 2))
        
        self.process_scores_with_percentile()
        
        return prediction 
    
    
    def decision_function(self,X,y):
        """
         X : numpy array of shape (n_samples, n_features)

         Returns  anomaly scores : numpy array of shape (n_samples,)
                 The anomaly score of the input samples.
        """

        # Predict X and return reconstruction errors
        prediction = self.model.predict(X)
        return (np.linalg.norm(y - prediction, axis = 2))

    def load_model(self, model_path=""):
        self.model.load_weights(model_path if model_path else self.model_path)

    def save_model(self, model_path=""):
        if not model_path and not self.model_path:
            raise Exception("You must provide a path to save model")
        self.model.save_weights(model_path if model_path else self.model_path)

    



