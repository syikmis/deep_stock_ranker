import os

import bcolors
from scipy.stats import pearsonr

import model.data_loader as dl
import model.web_scrapper as ws
from model.model import *
from model.utils import *

EXPERIMENTS_PCC_ = "experiments/pccV2/"
RESULT_CSV = "experiments/ranking_resultV2.csv"
MODELS_PATH = "experiments/modelsV2/"
MODELS_SUFFIX = "_rankerV2.h5"

n_steps = 60

sectors = {"ADS.DE": ["clothing"],
           "ALV.DE": ["insurance", ],
           "BAS.DE": ["chemistry", "industry"],
           "BAYN.DE": ["chemistry", "pharma"],
           "BEI.DE": ["chemistry", ],
           "BMW.DE": ["automotive", ],
           "CON.DE": ["automotive", ],
           "1COV.DE": ["chemistry", ],
           "DAI.DE": ["automotive", ],
           "DBK.DE": ["finance", ],
           "DB1.DE": ["finance", ],
           "LHA.DE": ["aviation", ],
           "DPW.DE": ["logistics", "industry"],
           "DTE.DE": ["technology", ],
           "EOAN.DE": ["provider", ],
           "FRE.DE": ["medical", ],
           "FME.DE": ["medical", ],
           "HEI.DE": ["real_estate", "industry"],
           "HEN3.DE": ["consumer", "chemistry"],
           "IFX.DE": ["technology"],
           "LIN.DE": ["industry"],
           "MRK.DE": ["chemistry", "pharma"],
           "MTX.DE": ["aviation"],
           "MUV2.DE": ["insurance"],
           "RWE.DE": ["provider"],
           "SAP.DE": ["technology"],
           "SIE.DE": ["technology"],
           "VOW3.DE": ["automotive"],
           "VNA.DE": ["real_estate"],
           "WDI.DE": ["technology", "finance"]}


def save_pccs(models_pccs):
    if not os.path.isdir(EXPERIMENTS_PCC_):
        os.mkdir(EXPERIMENTS_PCC_)
    for key in list(models_pccs.keys()):

        with open(EXPERIMENTS_PCC_ + key + ".csv", "w") as dst:
            dst.write("NO,TICKER,PCC\n")
            values = list(models_pccs[key])
            for i, x in list(enumerate(values)):
                dst.write("{},{},{}\n".format(i, x[0], x[1]))


def save_rankings(rankings):
    with open(RESULT_CSV, "w") as dst:
        dst.write("TICKER,RANK\n")
        for x, y in rankings.items():
            print(bcolors.BOLD + "Ticker: {} Rank {}".format(x, str(round(float(y), 10))) + bcolors.END)
            dst.write("{},{}\n".format(x, str(round(float(y), 10))))


def refresh_data():
    refresh = str(input("Do you want to refresh the data [test]? [y|N]: "))
    while refresh != "y" and refresh != "N":
        refresh = str(input("Do you want to refresh the data [test]? [y|N]: "))
    if refresh == "y":
        dl.refresh_dax_data()


def select_models_by_sectors(ticker):
    ticker_sectors = sectors[ticker]
    models = [x for x, y in sectors.items() if len(list(set(y) & set(ticker_sectors))) != 0]
    models = [x[:-3] + MODELS_SUFFIX for x in models]

    to_ignore = ["FRE_rankerV2.h5", "1COV_rankerV2.h5"]
    models = set(models)
    to_ignore = set(to_ignore)
    models.difference_update(to_ignore)
    return list(models)


def predict(selected):
    refresh_data()
    rankings = {}
    models_pccs = {}
    models = os.listdir(MODELS_PATH)
    models = [x for x in models if not x.startswith(".")]
    tickers = ws.get_tickers()
    i = 0
    for model in models:
        key = model.rstrip(".h5") + "_pcc"
        models_pccs[key] = []

    for ticker in tickers:
        print(bcolors.OKMSG + bcolors.BLUEIC + "[INFO] Compute ranking value for " + ticker + bcolors.END)
        test_x, test_y = dl.get_test_data(ticker, n_steps=n_steps)

        n_features = test_x[0].shape[1]
        rank_avg_model = []
        if selected:
            models = select_models_by_sectors(ticker)
            if not models:
                print(bcolors.WARN + "[WARN] No suited models found for " + ticker + " -> passing.." + bcolors.END)
                continue
        for model_name in models:
            key = model_name.rstrip(".h5") + "_pcc"
            model = SpookyArtificialIntelligenceV2(n_steps, n_features).get_model()
            try:
                model.load_weights(MODELS_PATH + model_name)
            except OSError:
                print(bcolors.ERR + "No .h5 found for " + model_name + "\n>>>still continuing..." + bcolors.END)
                continue
            # yhat is list of predicted mean return values for next 10 days for each sample of 60 days
            yhat = model.predict(test_x)
            # pearson correlation coefficient between predicted rank value and log(stock_return)
            pcc = pearsonr(test_y, yhat)
            # collect pcc for all stocks for each model
            models_pccs[key].append((ticker, pcc[0].item()))
            yhat = moving_average(yhat, 10)
            # rank per stock & each model := mean(mvg_avg10(predicted_rank)
            rank_avg_model.append(np.mean(yhat))
        # rank per stock & over all models
        rank_avg_overall = np.mean(rank_avg_model)
        rankings[ticker] = rank_avg_overall
        i += 1
        print(bcolors.OKMSG + "[INFO] {0:.0%} completed".format(i / len(tickers)) + bcolors.END)
        print(bcolors.OKMSG + RESULT_CSV + bcolors.END)

    save_rankings(rankings)
    save_pccs(models_pccs)


if __name__ == "__main__":
    opening = "Welcome to \"Deep Stock Ranker\":\n" \
              "A LSTM Neural Network Model for Stock Selection"
    print(bcolors.BLUE + bcolors.BOLD + bcolors.UNDERLINE + "{:*^30}".format(opening) + bcolors.END)
    selection = str(input("Do you want to predict stocks by selected models only (same sector)? [y|N]: "))
    if selection == "y":
        selection = True
    else:
        selection = False
    predict(selected=selection)
