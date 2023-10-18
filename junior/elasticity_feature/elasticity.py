import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import r2_score


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_skus = df.sku.unique()

    r2_scores = {}
    for sku in unique_skus:
        sku_df = df[df.sku == sku]
        prices = sku_df.price.values
        log_qty = np.log1p(sku_df.qty)

        res = linregress(prices, log_qty)
        elasticity = res.rvalue**2

        r2_scores[sku] = elasticity

    tuples = list(r2_scores.items())
    out_df = pd.DataFrame(tuples, columns=["sku", "elasticity"])

    return out_df
