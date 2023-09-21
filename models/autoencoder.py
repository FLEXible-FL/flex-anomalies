from keras.models import Sequential
from keras.layers import Dense
from models import BaseModel

class AutoEncoder(BaseModel):

    """
    neurons : list, The number of neurons per hidden layers.

    hidden_act : str or list, optional default='relu'
         All hidden layers do not necessarily have to use the same activation type.

    output_act: str, optional (default='linear'),output activation of the final laye


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
    """

    def __init__(
        self,
        input_dim,
        neurons,
        callbacks,
        hidden_act="relu",
        output_act="linear",
        loss="mse",
        validation_size=0.2,
        batch_size=32,
        epoch=1,
        optimizer="adam",
    ) -> None:
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.callbacks = callbacks
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
        self.epoch = epoch
        self.optimizer = optimizer
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(
            Dense(
                self.neurons[0],
                activation=self.hidden_act[0],
                input_shape=(self.input_dim,),
            )
        )

        # Additional layers
        for neurons, hidden_act in zip(self.neurons[1:], self.hidden_act[1:]):
            model.add(Dense(neurons, activation=hidden_act))

        # Output layers
        model.add(Dense(self.input_dim, activation=self.output_act))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        print(model.summary())
        return model

    def fit(self, X, y=None):
        """
        X : numpy array of shape (samples, features)
        y:  Ignored in unsupervised methods

        """
        self.history = self.model.fit(
            X,
            X,
            epochs=self.epoch,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_size,
            verbose=1,
        ).history

    def predict_outlier(self, X):
        pred = self.model.predict(X)
        # scores = metric(data,pred)
        # process_scores

    def evaluate(self, X):
        pass

    def load_pretrained_model(self, model_path="./pretrained/autoencoder"):
        self.model.load_weights(model_path)

    def save_model(self, model_path="./pretrained/autoencoder"):
        self.model.save_weights(model_path)
