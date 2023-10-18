import itertools
from typing import Dict, List, Tuple

import numpy as np


class SimilarItems:
    @staticmethod
    def similarity(
        embeddings: Dict[int, np.ndarray]
    ) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """

        indices = list(embeddings.keys())

        pair_sims = dict()

        for pair in itertools.combinations(indices, 2):
            left_idx, right_idx = pair
            a = embeddings[left_idx]
            b = embeddings[right_idx]
            cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            pair_sims[(left_idx, right_idx)] = round(cosine, 8)

        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float],
        top: int,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        nearest_neighbors = {}

        for pair, sim in sim.items():
            a, b = pair

            if a not in nearest_neighbors:
                nearest_neighbors[a] = list()
            nearest_neighbors[a].append((b, sim))

            if b not in nearest_neighbors:
                nearest_neighbors[b] = list()
            nearest_neighbors[b].append((a, sim))

        for k, v in nearest_neighbors.items():
            nearest_neighbors[k] = sorted(
                nearest_neighbors[k], key=lambda x: -x[1]
            )[:top]

        return nearest_neighbors

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = dict()

        for k, neighbors in knn_dict.items():
            neighbor_prices = np.array([prices[i] for i, _ in neighbors])
            neighbor_sim = np.array([cosine + 1 for _, cosine in neighbors])
            neighbor_sim = neighbor_sim / np.sum(neighbor_sim)

            weighted_prices = neighbor_prices * neighbor_sim
            price = round(np.sum(weighted_prices), 2)
            knn_price_dict[k] = price

        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """

        sims = SimilarItems.similarity(embeddings)
        knn = SimilarItems.knn(sims, top=top)
        knn_price_dict = SimilarItems.knn_price(knn, prices)
        return knn_price_dict
