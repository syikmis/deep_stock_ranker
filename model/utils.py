import numpy as np


def compute_features(df):
    columns = list(df.columns.values)

    for column in columns:
        df[column + "_rtn"] = np.log(df[column]) - np.log(df[column].shift(1))
        df.drop([column], 1, inplace=True)

    return clean_dataframe(df)


def clean_dataframe(df):
    df.fillna(0, inplace=True)
    # df = df[np.isfinite(df).all(1)]
    return df


# split a multivariate sequence into samples
def split_dataframe(df, n_steps=60):
    # columns = list(df.columns.values)
    X, y = list(), list()
    for i in range(df.shape[0]):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(df) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = df.iloc[i:end_ix, :]
        seq_y = df.iloc[end_ix, :]
        X.append(seq_x.values)
        y.append(seq_y.item())

    X = np.array(X)
    y = np.array(y)
    return X, y


def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def r_chop(string, ending):
    if string.endswith(ending):
        return string[:-len(ending)]
    return string
