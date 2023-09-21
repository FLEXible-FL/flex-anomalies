from keras.models import Sequential
from keras.layers import Dense


class AutoEncoder:
    def __init__(
        self,
        input_dim,
        neurons,
        callbacks,
        hidden_act="relu",
        output_act="linear",
        validation=0.2,
        batch=32,
        apoch=1,
        optimizer="adam",
    ) -> None:
        self.input_dim = input_dim
        self.neurons = neurons
        self.callbacks = callbacks
        self.hidden_act = (
            list(hidden_act) * len(neurons)
            if type(hidden_act) is not list
            else hidden_act
        )
        self.hidden_act = (
            self.hidden_act
            + [self.hidden_act[-1]] * (len(neurons) - len(self.hidden_act))
            if len(self.hidden_act) < len(self.neurons)
            else self.hidden_act
        )
        self.output_act = output_act
        self.validation = validation
        self.batch = batch
        self.apoch = apoch
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
        for neurons, hidden_act in zip(self.neurons[1:],self.hidden_act[1:]):
            model.add(
                Dense(
                    neurons,
                    activation=hidden_act
                )
            )

        # Output layers
        model.add(
            Dense(
                self.input_dim,
                activation=self.output_act
            )
        )

        # Compile model
        model.compile(loss='mse', optimizer=self.optimizer)
        print(model.summary())
        return model

    def fit(self, data):
        pass

    def predict_outlier(self, data, threshold = 0.75):
        mse = 0.8
        prediction = mse>threshold
        return prediction, mse

    def evaluate(self, data):
        pass

    def load_pretrained_model(self, model_path="./pretrained/autoencoder"):
        self.model.load_weights(model_path)

    def save_model(self, model_path="./pretrained/autoencoder"):
        self.model.save_weights(model_path)

