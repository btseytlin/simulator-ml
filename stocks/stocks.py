import numpy as np
import pandas as pd


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    forecast_items = np.floor(df["gmv"] / df["price"])
    forecast_items = np.minimum(forecast_items, df["stock"])
    df["gmv"] = forecast_items * df["price"]

    # def postprocess(row):
    #     gmv = row.gmv
    #     price = row.price
    #     stock = row.stock
    #     forecast_items = np.floor(gmv / price)

    #     forecast_items = min(forecast_items, stock)

    #     return forecast_items * price

    # df["gmv"] = df.apply(postprocess, axis=1)
    return df


data = {
    "sku": [100, 200, 300],
    "gmv": [400, 350, 500],
    "price": [100, 70, 120],
    "stock": [3, 10, 5],
}

df = limit_gmv(pd.DataFrame(data))
print(df)
