import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, cast

import numpy as np
import umap
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from umap.umap_ import nearest_neighbors

RANDOM_SEED = 42
DATASETS: list[str] = ["iris", "digits", "wine"]

NN_VALUES = [5, 10, 15, 30, 50]

RECTANGLES_PER_PROJECTION = 500

CURRENT_DIR = Path(__file__).parent
DATA_DIR = Path(CURRENT_DIR, "data")


def load_dataset(name: str) -> tuple[NDArray, NDArray, dict]:
    if name == "iris":
        d = datasets.load_iris()
    elif name == "digits":
        d = datasets.load_digits()
    elif name == "wine":
        d = datasets.load_wine()
    else:
        raise ValueError(f"Unknown dataset: {name}")

    d = cast(Bunch, d)
    X = d.data
    y = d.target
    meta = {"feature_names": d.feature_names}

    return X, y, meta


def compute_precomputed_knn(X: NDArray, n_neighbors: int, random_state: int):
    knn_indices, knn_dists, _ = nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        metric_kwds=None,
        angular=False,
        random_state=random_state,
    )
    return knn_indices, knn_dists


def compute_umap(
    X: np.ndarray,
    n_neighbors: int,
    random_state: int = 0,
    knn_indices=None,
    knn_dists=None,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=2, random_state=random_state
    )
    if (knn_indices is not None) and (knn_dists is not None):
        embedding = reducer.fit_transform(
            X, knn_indices=knn_indices, knn_dists=knn_dists
        )
    else:
        embedding = reducer.fit_transform(X)

    embedding = cast(NDArray, embedding)

    return embedding


def sample_rectangles(embedding: NDArray, n_rects: int) -> List[Dict]:
    xs = embedding[:, 0]
    ys = embedding[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    eps = np.finfo(float).eps

    rects = []
    for _ in range(n_rects):
        cx = random.uniform(xmin, xmax)
        cy = random.uniform(ymin, ymax)

        max_half_width = max(eps, min(cx - xmin, xmax - cx))
        max_half_height = max(eps, min(cy - ymin, ymax - cy))
        half_w = random.uniform(0.0, max_half_width)
        half_h = random.uniform(0.0, max_half_height)

        x0, x1 = cx - half_w, cx + half_w
        y0, y1 = cy - half_h, cy + half_h

        w = 2.0 * half_w
        h = 2.0 * half_h

        inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        indices = np.nonzero(inside)[0]
        rects.append(
            {
                "center": (cx, cy),
                "width": float(w),
                "height": float(h),
                "bounds": (float(x0), float(y0), float(x1), float(y1)),
                "indices": indices.tolist(),
            }
        )

    return rects


def save_result(path: Path, payload: Dict):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for ds_name in DATASETS:
        print(f"Processing dataset: {ds_name}")
        X, y, meta = load_dataset(ds_name)

        X = StandardScaler().fit_transform(X)

        for nn in NN_VALUES:
            print(f" - computing KNN (n_neighbors={nn}) and UMAP")
            knn_indices, knn_dists = compute_precomputed_knn(
                X, n_neighbors=nn, random_state=RANDOM_SEED
            )

            embedding = compute_umap(
                X,
                n_neighbors=nn,
                random_state=RANDOM_SEED,
                knn_indices=knn_indices,
                knn_dists=knn_dists,
            )

            rects = sample_rectangles(embedding, RECTANGLES_PER_PROJECTION)

            payload = {
                "dataset_name": ds_name,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "y": y.tolist(),
                "X": X,
                "meta": meta,
                "umap_n_neighbors": int(nn),
                "umap_random_state": int(RANDOM_SEED),
                "embedding": embedding.astype(np.float32),
                "rectangles": rects,
                "created_at": time.time(),
            }

            out_path = Path(DATA_DIR, f"{ds_name}_umap_nn_{nn}.pkl")
            save_result(out_path, payload)
            print(f"   saved -> {out_path}")


if __name__ == "__main__":
    main()
