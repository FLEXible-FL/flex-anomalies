#Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


from keras.models import Sequential
from keras.layers import Dense
from flexanomalies.utils import BaseModel
from sklearn.preprocessing import StandardScaler
import numpy as np
from flexanomalies.utils.metrics import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class AutoEncoder(BaseModel):

    """
    neurons : list, The number of neurons per hidden layers.

    hidden_act : str or list, optional default='relu'
         All hidden layers do not necessarily have to use the same activation type.

    output_act: str, optional (default='linear'),output activation of the final layer


    loss : str or obj, optional (default= 'mse')
        Name of objective function or objective function.


    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.


    epochs : int, optional (default=1)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    validation_size : float, optional (default=0.2)
        The percentage of data to be used for validation.

    callbacks: list, tensorflow callbacks

    input_dim : int, number of features

    contamination : float in (0., 0.5), optional (default=0.1)
         Contamination of the data set, the proportion of outliers in the data set.


    """

    def __init__(
        self,
        input_dim,
        neurons,
        model_path="",
        callbacks=[],
        hidden_act="relu",
        output_act="linear",
        loss="mse",
        validation_size=0.2,
        batch_size=32,
        epochs=1,
        optimizer="adam",
        contamination=0.1,
        w_size=None,
        n_pred=1,
        preprocess=True,
    ) -> None:
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.input_dim = input_dim
        self.neurons = neurons
        self.model_path = model_path
        self.w_size = w_size
        self.n_pred = n_pred
        self.callbacks = []
        self.update_callbacks(callbacks=callbacks)

        self.hidden_act = (
            list(hidden_act) * len(neurons)
            if type(hidden_act) is not list
            else hidden_act
        )
        # Validate and complete the number of hidden neurons
        self.hidden_act = (
            self.hidden_act
            + [self.hidden_act[-1]] * (len(neurons) - len(self.hidden_act))
            if len(self.hidden_act) < len(self.neurons)
            else self.hidden_act
        )
        self.output_act = output_act
        self.loss = loss
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.preprocess = preprocess
        self.scaler = StandardScaler()

        self.model = self._build_model()

    def update_callbacks(self, callbacks):
        model_checkpoint_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.005
        )
        self.callbacks = [model_checkpoint_callback] + callbacks

    def _build_model(self):

        model = Sequential()
        
        if self.w_size is None:
            input_shape = (self.input_dim,)
        else:
            input_shape = (self.w_size, self.input_dim)

        # Input layer
        model.add(
            Dense(
                self.neurons[0],
                activation=self.hidden_act[0],
                input_shape=input_shape,
            )
        )

        # Additional layers
        for neurons, hidden_act in zip(self.neurons[1:], self.hidden_act[1:]):
            model.add(Dense(neurons, activation=hidden_act))

        # Output layers
        model.add(Dense(self.input_dim, activation=self.output_act))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods

        """

        if self.preprocess:
            Xscaler = self.scaler.fit_transform(X)
        else:
            Xscaler = np.copy(X)

        np.random.shuffle(Xscaler)

        self.history_ = self.model.fit(
            Xscaler,
            Xscaler,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_size,
            verbose=1,
            callbacks=self.callbacks,
        ).history
        # self.model.load_weights(self.model_path)

    def predict(self, X, y=None):
        if self.preprocess:
            Xscaler = self.scaler.fit_transform(X)
        else:
            Xscaler = np.copy(X)

        prediction_scores = self.model.predict(Xscaler)
        self.d_scores_ = distances(Xscaler, prediction_scores)

        self.process_scores()

        return prediction_scores

    def decision_function(self, X, y=None):
        """
        X : numpy array of shape (n_samples, n_features)

        Returns  anomaly scores : numpy array of shape (n_samples,)
                The anomaly score of the input samples.
        """

        if self.preprocess:
            Xscaler = self.scaler.transform(X)
        else:
            Xscaler = np.copy(X)

        # Predict X and return reconstruction errors
        prediction_scores = self.model.predict(Xscaler)
        return distances(Xscaler, prediction_scores)

    def load_model(self, model_path=""):
        self.model.load_weights(model_path if model_path else self.model_path)

    def save_model(self, model_path=""):
        if not model_path and not self.model_path:
            raise Exception("You must provide a path to save model")
        self.model.save_weights(model_path if model_path else self.model_path)
