import os

import bcolors

import model.data_loader as dl
import model.web_scrapper as ws
from model.model import SpookyArtificialIntelligenceV1, SpookyArtificialIntelligenceV2

n_steps = 60
n_outlook = 1
models_path = "experiments/modelsV1/"
model_suffix = "_rankerV1.h5"


def refresh_data():
    reload = str(input("Do you want to reload the data (train, val)? [y|N]: "))
    while reload != "y" and reload != "N":
        reload = str(input("Do you want to reload the data (train, val)? [y|N]: "))

    if reload == "y":
        dl.reload_train_val_data(overwrite=True)


def train():
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
    refresh_data()
    tickers = ws.get_tickers()
    #tickers = ["EOAN.DE"]

    for ticker in tickers:
        mes = "Build model for " + ws.ticker_to_name(ticker)
        print(bcolors.BLUE + bcolors.BOLD + bcolors.UNDERLINE + "{:*^30}".format(mes) + bcolors.END)
        train_x, train_y, val_x, val_y = dl.get_train_data(ticker, n_steps)
        n_featues = train_x[0].shape[1]
        model = SpookyArtificialIntelligenceV1(n_steps, n_featues).get_model()
        model.fit(x=train_x,
                  y=train_y,
                  validation_data=(val_x, val_y),
                  epochs=20,
                  verbose=1,
                  shuffle=False)
        model_name = ticker[:-3] + model_suffix
        model.save_weights(models_path + model_name)
        print(bcolors.OK + "[INFO] Saved " + model_name + bcolors.END)

if __name__ == "__main__":
    opening = "Welcome to \"Deep Stock Ranker\":\n" \
              "A LSTM Neural Network Model for Stock Selection"
    print(bcolors.BLUE + bcolors.BOLD + bcolors.UNDERLINE + "{:*^30}".format(opening) + bcolors.END)
    train()
