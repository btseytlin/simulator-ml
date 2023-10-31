import os
from typing import Tuple

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from sklearn.neighbors import KernelDensity

DIVERSITY_THRESHOLD = 10

app = FastAPI()
embeddings = {}


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """

    model = KernelDensity()
    model.fit(embeddings)
    scores = model.score_samples(embeddings)
    uniqueness_scores = 1 / np.exp(scores)
    return uniqueness_scores


def group_diversity(
    embeddings: np.ndarray, threshold: float
) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    uniqueness_scores = kde_uniqueness(embeddings)
    diversity_score = np.mean(uniqueness_scores)
    reject = diversity_score < threshold
    return reject, diversity_score


@app.on_event("startup")
@repeat_every(seconds=10)
def load_embeddings() -> dict:
    """Load embeddings from file."""

    # Load new embeddings each 10 seconds
    path = os.path.join(os.path.dirname(__file__), "embeddings.npy")
    embeddings_raw = np.load(path, allow_pickle=True).item()
    for item_id, embedding in embeddings_raw.items():
        embeddings[item_id] = embedding

    return {}


@app.get("/uniqueness/")
def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    item_embeddings = np.array([embeddings[id_] for id_ in item_ids])
    if item_embeddings.size != 0:
        uniqueness_scores = list(kde_uniqueness(item_embeddings))
        for id_, score in zip(item_ids, uniqueness_scores):
            item_uniqueness[id_] = float(score)

    return item_uniqueness


@app.get("/diversity/")
def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    answer = {"diversity": 0.0, "reject": True}

    # Calculate diversity
    item_embeddings = np.array([embeddings[id_] for id_ in item_ids])

    if item_embeddings.size != 0:
        reject, diversity_score = group_diversity(
            item_embeddings,
            threshold=DIVERSITY_THRESHOLD,
        )
        answer["diversity"] = float(diversity_score)
        answer["reject"] = bool(reject)

    return answer


def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=5000)


if __name__ == "__main__":
    main()
