import numpy as np
import pandas as pd


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    df = df.copy()
    no_nans = df.dropna(subset=[target])
    group_means = (
        no_nans.groupby(group).mean()[target].apply(lambda x: np.floor(x))
    )

    df[target] = df.apply(
        lambda x: group_means.loc[x[group]]
        if np.isnan(x[target])
        else x[target],
        axis=1,
    )
    return df
