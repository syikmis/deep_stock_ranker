import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm as cm

import model.data_loader as dl
import model.web_scrapper as ws
from model.utils import *

RESULT_CSV = "experiments/ranking_resultV1_SELECTED.csv"
PCC_DIR = "experiments/pccV1/"
PCC_FILE = "_rankerV1_pcc.csv"
RANK = "RANK"


def evaluate(n_stocks):
    fig_name = "experiments/plots/prt_rtn_V2_SELECTED_TOP_" + str(n_stocks) + ".jpeg"

    df = pd.read_csv(RESULT_CSV, sep=",")
    # df[RANK] = df[RANK] * -1
    df.sort_values(RANK, ascending=False, inplace=True)
    df = df.iloc[:n_stocks, ]
    stocks = list(df.TICKER)
    names = list()
    main_df = pd.DataFrame()
    print("Selected stocks: ")
    for stock in stocks:
        print(ws.ticker_to_name(stock))
        names.append(ws.ticker_to_name(stock))
        # df_val = dl.compute_com_df(stock, "val")
        df = dl.compute_com_df(stock, "test")
        # df = df_val.append(df_test)
        if main_df.empty:
            main_df = df
        else:
            main_df = pd.concat([main_df, df], axis=1, sort=False)
    df = main_df
    df_sum = df.sum(axis=1)
    first = df_sum.iloc[0]
    last = df_sum.iloc[-1]
    total_return = (last - first) / first
    # print(sum.head())

    fig, ax = plt.subplots()
    x = np.array(df_sum.index)
    y = np.array(df_sum)
    ax.plot(x, y, color="red", linewidth=2, alpha=0.7)
    for i, col in list(enumerate(list(main_df.columns.values))):
        ax.plot(x, df[col], linewidth=1, alpha=0.4, label=r_chop(col, "_adj_close"))
    ax.set(xlabel='date [d]', ylabel='Adj Close of Portfolio [€]',
           title='Portfolio TOP' + str(n_stocks) + ' Return: {0:.2%}'.format(total_return))
    ax.tick_params(axis="x", labelsize=8, labelcolor="black")
    ax.tick_params(axis="y", labelsize=16, labelcolor="black")
    ax.set_xticks(ax.get_xticks()[::30])
    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)
        label.set_horizontalalignment("right")

    plt.text(0.05, 0.95, "\n".join(names),
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.subplots_adjust(left=0.15)
    fig.savefig(fig_name)
    return plt


def create_pcc_df():
    tickers = ws.get_tickers()
    df = pd.DataFrame(columns=tickers, index=tickers)
    df = df.fillna(0)

    pcc_csv = os.listdir(PCC_DIR)

    for csv in pcc_csv:
        key = csv[:-len(PCC_FILE)]
        df_temp = pd.read_csv(PCC_DIR + csv)
        df_temp = df_temp.fillna(0)

        values = list(df_temp["PCC"].round(20).values.flatten())
        df[key] = values
    temp = []
    for ticker in tickers:
        temp.append(ws.ticker_to_name(ticker))

    tickers = temp
    return df, tickers


def plot_correlation_matrix():
    df, tickers = create_pcc_df()
    data = df.values
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(df.values, cmap=plt.cm.RdYlGn)
    cmap = cm.get_cmap()
    plt.title('Predictor Correlation')
    ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(tickers, fontsize=6)
    ax.set_yticklabels(tickers, fontsize=6)

    ax.tick_params(axis="both", which="major", pad=10, colors="black")
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    # plt.tight_layout()
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(heatmap, ticks=[-1, -.5, 0, .5, 1, 1])
    plt.show()


def print_top_plots():
    evaluate(3)
    evaluate(5)
    evaluate(10)


if __name__ == "__main__":
    print_top_plots()
    plot_correlation_matrix()
