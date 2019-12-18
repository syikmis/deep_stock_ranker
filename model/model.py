from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.models import Sequential


class SpookyArtificialIntelligence:
    # architecture from paper
    def __init__(self, n_steps, n_features):
        self.model = self.build_model(n_steps, n_features)

    def build_model(self, n_steps, n_features):
        model = Sequential()
        model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(n_features, activation="tanh"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_model(self):
        return self.model


class SpookyArtificialIntelligenceV2:

    def __init__(self, n_steps, n_features):
        self.model = self.build_model(n_steps, n_features)

    def build_model(self, n_steps, n_features):
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(Flatten())
        model.add(Dense(30, activation='relu'))
        model.add(Dense(n_features, activation="tanh"))
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_model(self):
        return self.model
