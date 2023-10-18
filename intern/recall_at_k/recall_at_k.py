from typing import List

import numpy as np


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    labels = np.array(labels)
    scores = np.array(scores)

    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]

    relevant_items = sum(labels[i] for i in top_k_indices)

    recall_at_k = relevant_items / sum(labels)

    return recall_at_k


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    labels = np.array(labels)
    scores = np.array(scores)

    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]

    top_k_labels = labels[top_k_indices]

    precision_at_k = np.sum(top_k_labels) / k
    return precision_at_k


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    labels = np.array(labels)
    scores = np.array(scores)

    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]
    bottom_k_indices = sorted_indices[k:]

    true_negatives = sum(1 - labels[i] for i in bottom_k_indices)

    false_positives = sum(1 - labels[i] for i in top_k_indices)

    total_negatives = true_negatives + false_positives

    if total_negatives == 0:
        specificity_at_k = 0
    else:
        specificity_at_k = true_negatives / total_negatives

    return specificity_at_k


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    precision = precision_at_k(labels, scores, k)
    recall = recall_at_k(labels, scores, k)

    f1 = (2 * precision * recall) / (precision + recall + 1e-10)
    return f1
