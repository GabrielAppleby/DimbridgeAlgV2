from typing import cast

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.utils import Bunch


def get_cali_housing_example() -> tuple[np.ndarray, np.ndarray]:
    housing = cast(Bunch, fetch_california_housing(as_frame=True))
    df = housing["data"]

    num_selections = 5
    selections = np.zeros([num_selections, df.shape[0]]).astype(np.bool)
    selections[0, df["AveRooms"] < 6.0] = 1
    selections[1, df["AveBedrms"] < 1.3] = 1
    selections[2, df["Population"] < 400.0] = 1
    selections[3, df["AveOccup"] < 2.3] = 1
    selections[4, df["MedInc"] < 5.0] = 1

    return df.to_numpy(dtype=np.float32), selections
