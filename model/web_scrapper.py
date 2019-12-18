import datetime as dt
import os
import pickle
import time
from pathlib import Path

import bcolors
import bs4 as bs
import pandas_datareader.data as web
import requests

from model import data_loader as dl

DATA_DIR = Path("data")
COM_DATA_DIR = DATA_DIR / "DAX30"
TRAIN_DIR = COM_DATA_DIR / "train"
VAL_DIR = COM_DATA_DIR / "val"
TEST_DIR = COM_DATA_DIR / "test"
PKL_DIR = DATA_DIR / "PKL_DIR"
COM_NAMES_PKL = PKL_DIR / "DAX30_names.pkl"
COM_TICKERS_PKL = PKL_DIR / "DAX30_tickers.pkl"


def path_to_string(path):
    return path.as_posix()


def get_com_tickers_names():
    resp = requests.get(
        "https://de.wikipedia.org/wiki/DAX#Unternehmen_im_DAX")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    names = []
    for row in table.findAll("tr")[1:]:
        name = row.findAll("td")[0].text
        ticker = row.findAll("td")[1].text
        tickers.append(ticker.rstrip() + ".DE")
        names.append(name.rstrip())

    save_names(names)
    save_tickers(tickers)


def get_com_data(times, ticker=None):
    """
    Loads stock data from yahoo and saves it as .csv for each company
    """
    paths = [COM_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR]
    for path in paths:
        temp = path_to_string(path)
        if not os.path.exists(temp):
            os.makedirs(temp)

    if ticker is None:
        tickers = get_tickers()
    else:
        tickers = [ticker]

    for mode, start, end in times:

        print(bcolors.WAITMSG + "[INFO] Load " + mode + " data from " + start.strftime("%d/%m/%Y") + " till " +
              end.strftime("%d/%m/%Y") + bcolors.END)
        for ticker in tickers:
            if mode is not "test":
                file = ticker + ".csv"
                file = COM_DATA_DIR / mode / file
                if not file.is_file():
                    # just in case your connection breaks, we"d like to save our progress!
                    print("[INFO] Fetch data for " + ticker + " ...")
                    try:
                        df = web.DataReader(ticker, "yahoo", start, end)
                    except (IOError, KeyError):
                        print(bcolors.WARN + "No data from yahoo found for: {}".format(ticker) + bcolors.END)
                        continue
                    dl.save_com_as_csv(df, ticker, mode)
                    time.sleep(10)
                else:
                    print("[INFO] Skipped loading " + ticker + "! Already saved!")
                    continue
            else:
                try:
                    df = web.DataReader(ticker, "yahoo", start, end)
                except (IOError, KeyError):
                    print(bcolors.WARN + "No data from yahoo found for: {}".format(ticker) + bcolors.END)
                    continue
                dl.save_com_as_csv(df, ticker, mode)
                time.sleep(10)


def get_tickers():
    """
    Returns list with all 30 DAX company tickers
    :return: list with ticker symbols
    """
    with open(path_to_string(COM_TICKERS_PKL), "rb") as f:
        tickers = pickle.load(f)

    return tickers


def get_names():
    """
    Returns list with all 30 DAX company names
    :return: list with all DAX company names
    """
    with open(path_to_string(COM_NAMES_PKL), "rb")as f:
        names = pickle.load(f)
    return names


def save_tickers(tickers):
    path = path_to_string(PKL_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_to_string(COM_TICKERS_PKL), "wb") as f:
        pickle.dump(tickers, f)
        print("[INFO] Saved DAX 30 ticker symbols")


def save_names(names):
    path = path_to_string(PKL_DIR)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path_to_string(COM_NAMES_PKL), "wb") as f:
        pickle.dump(names, f)
        print("[INFO] Saved DAX 30 company names")


def ticker_to_name(ticker):
    names = list(get_names())
    tickers = list(get_tickers())
    index = tickers.index(ticker)
    name = names.pop(index)
    return name


if __name__ == "__main__":
    train_start = dt.datetime(2005, 1, 1)
    train_end = dt.datetime(2015, 12, 31)
    df = web.DataReader("1COV.DE", "yahoo", train_start, train_end)
