import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model.data_loader as dl
import model.web_scrapper as ws

RESULT_CSV = "experiments/ranking_resultV2.csv"
RANK = "RANK"


def evaluate(n_stocks):
    FIG_NAME = "experiments/plots/portfolio_return_V2_TOP_" + str(n_stocks) + ".jpeg"

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
    sum = df.sum(axis=1)
    first = sum.iloc[0]
    last = sum.iloc[-1]
    total_return = (last - first) / first
    # print(sum.head())

    fig, ax = plt.subplots()
    x = np.array(sum.index)
    y = np.array(sum)
    ax.plot(x, y, color="red", linewidth=2, alpha=0.7)
    for i, col in list(enumerate(list(main_df.columns.values))):
        ax.plot(x, df[col], linewidth=1, alpha=0.4, label=col.rstrip("_adj_close"))
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
    fig.savefig(FIG_NAME)
    return plt


def print_top_plots():
    evaluate(3)
    evaluate(5)
    evaluate(10)

if __name__ == "__main__":
    print_top_plots()
