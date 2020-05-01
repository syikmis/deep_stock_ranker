import datetime as dt
import pickle
from pathlib import Path

import bcolors
import pandas as pd

import model.web_scrapper as ws
from model.utils import *

DATA_DIR = Path("data")
COM_DATA_DIR = DATA_DIR / "DAX30"
TRAIN_DIR = COM_DATA_DIR / "train"
VAL_DIR = COM_DATA_DIR / "val"
TEST_DIR = COM_DATA_DIR / "test"
PKL_DIR = DATA_DIR / "PKL_DIR"
TRAIN_PKL = PKL_DIR / "DAX30_train.pkl"
VAL_PKL = PKL_DIR / "DAX30_val.pkl"
TEST_PKL = PKL_DIR / "DAX30_test.pkl"

train_start = dt.datetime(1995, 1, 1)
train_end = dt.datetime(2018, 12, 31)
val_start = dt.datetime(2019, 1, 1)
val_end = dt.datetime(2020, 2, 29)
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
modes = ["train", "val", "test"]


def path_to_string(path):
    return path.as_posix()


def compute_dax_df(mode):
    """
    returns main dataframe with stocks data of all dax companies
    :return: dataframe with stocks data
    """

    tickers = ws.get_tickers()
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        ticker = ticker.rstrip()
        try:
            df = _get_com_as_df(ticker, mode)
        except IOError:
            print(bcolors.WARN +
                  "[INFO][" + mode + "] No .csv found for {}".format(ticker) + bcolors.END)
            continue
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df.drop(["index"], 1, inplace=True)
        columns = {"Adj Close": ticker.rstrip(".DE") + "_adj_close",
                   # "Open": ticker.rstrip(".DE") + "_open",
                   # "High": ticker.rstrip(".DE") + "_high",
                   # "Low": ticker.rstrip(".DE") + "_low",
                   # "Close": ticker.rstrip(".DE") + "_close",
                   # "Volume": ticker.rstrip(".DE") + "_vol"}
                   }
        df.rename(columns=columns, inplace=True)
        df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat([main_df, df], axis=1, sort=False)

    main_df.fillna(0, inplace=True)
    # print(main_df.tail())
    _save_dax_as_csv(main_df, mode)
    _save_dax_as_pkl(main_df, mode)
    return main_df


def compute_com_df(ticker, mode):
    df = pd.DataFrame()
    try:
        df = _get_com_as_df(ticker, mode)
    except IOError:
        print(bcolors.WARN +
              "[INFO][" + mode + "] No .csv found for {}".format(ticker) + bcolors.END)
        pass

    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    df.drop(["index"], 1, inplace=True)
    columns = {"Adj Close": r_chop(ticker, ".DE") + "_adj_close",
               # "Open": ticker.rstrip(".DE") + "_open",
               # "High": ticker.rstrip(".DE") + "_high",
               # "Low": ticker.rstrip(".DE") + "_low",
               # "Close": ticker.rstrip(".DE") + "_close",
               # "Volume": ticker.rstrip(".DE") + "_vol"}
               }
    df.rename(columns=columns, inplace=True)
    df.drop(["Open", "High", "Low", "Close", "Volume"], 1, inplace=True)
    df.fillna(0, inplace=True)
    return df


def save_com_as_csv(df, ticker, mode):
    """
    Saves company data as .csv

    :param df: new fetched dataframe from yahoo
    :param ticker: ticker symbol
    :param mode: mode for saving

    """
    if mode == "train":
        path = path_to_string(TRAIN_DIR) + "/{}.csv".format(ticker)
    elif mode == "val":
        path = path_to_string(VAL_DIR) + "/{}.csv".format(ticker)
    else:
        path = path_to_string(TEST_DIR) + "/{}.csv".format(ticker)

    df.to_csv(path)
    print("[INFO][" + mode + "] Saved {} data to {}".format(ticker, path))


def _save_dax_as_csv(df, mode):
    """
    Saves dax data as .csv
    param df: DataFrame

    """
    path = path_to_string(DATA_DIR) + "/DAX30_" + mode + ".csv"
    df.to_csv(path)
    print("[INFO][" + mode + "] Saved DAX30 data to {}".format(path))


def _save_dax_as_pkl(df, mode):
    path = path_to_string(PKL_DIR) + "/DAX30_" + mode + ".pkl"
    with open(path, "wb") as f:
        pickle.dump(df, f)
        print("[INFO][" + mode + "] Saved DAX30 data to {}".format(path))


def get_dax_as_pkl(mode):
    path = path_to_string(PKL_DIR) + "DAX30_" + mode + ".pkl"
    with open(path, "wb") as f:
        pickle.load(f)


def _get_com_as_df(ticker, mode):
    if mode == "train":
        path = path_to_string(TRAIN_DIR) + "/{}.csv".format(ticker)
    elif mode == "val":
        path = path_to_string(VAL_DIR) + "/{}.csv".format(ticker)
    else:
        path = path_to_string(TEST_DIR) + "/{}.csv".format(ticker)
    df = pd.read_csv(path)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return df


def reload_train_val_data(overwrite=False):
    times = [("train", train_start, train_end),
             ("val", val_start, val_end)]
    ws.get_com_tickers_names()
    # Fetch and save .csv for each com
    ws.get_com_data(times, overwrite=overwrite)
    print(bcolors.OKMSG + "[INFO] Reload data finished!" + bcolors.END)
    # compute Dataframe, save as pickle and return DataFrame
    for mode in ["train", "val"]:
        compute_dax_df(mode)


def reload_test_data():
    times = [("test", test_start, test_end)]
    # First get list from Wikipedia with all ticker symbols and name
    ws.get_com_tickers_names()
    # Fetch and save .csv for each com
    ws.get_com_data(times, overwrite=True)
    print(bcolors.OKMSG + "[INFO] Refresh data finished!" + bcolors.END)
    compute_dax_df("test")


def _get_com_train(ticker, force=False):
    if force:
        return compute_com_df(ticker, "train")
    else:
        if TRAIN_PKL.exists():
            return pd.read_pickle(path_to_string(TRAIN_PKL))
        else:
            return compute_com_df(ticker, "train")


def _get_com_val(ticker, force=False):
    if force:
        return compute_com_df(ticker, "val")
    else:
        if VAL_PKL.exists():
            return pd.read_pickle(path_to_string(VAL_PKL))
        else:
            return compute_com_df(ticker, "val")


def _get_com_test(ticker, force=False):
    if force:
        return compute_com_df(ticker, "test")
    else:
        if TEST_PKL.exists():
            return pd.read_pickle(path_to_string(TEST_PKL))
        else:
            return compute_com_df(ticker, "test")


def _load_train_val(ticker):
    train_df = _get_com_train(ticker, force=True)
    val_df = _get_com_val(ticker, force=True)
    return train_df, val_df


def get_train_data(ticker, n_steps):
    train_df, val_df = _load_train_val(ticker)
    train_df, val_df = compute_features(train_df), compute_features(val_df)
    train_x, train_y = split_dataframe(train_df, n_steps)
    val_x, val_y = split_dataframe(val_df, n_steps)
    return train_x, train_y, val_x, val_y


def get_test_data(ticker, n_steps):
    test_df = _get_com_test(ticker, force=True)
    test_df = compute_features(test_df)
    test_x, test_y = split_dataframe(test_df, n_steps=n_steps)
    test_y = np.expand_dims(test_y, axis=1)
    return test_x, test_y
