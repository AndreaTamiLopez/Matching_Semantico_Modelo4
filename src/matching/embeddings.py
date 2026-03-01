import numpy as np
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .utils import format_for_model


def build_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int,
    model_name: str,
    mode: str
):
    texts_for_emb = format_for_model(texts, mode=mode, model_name=model_name)
    return model.encode(
        texts_for_emb,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )


def knn_retrieve(
    E_pol,
    E_proy,
    top_n_candidates: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_neighbors = min(int(top_n_candidates), E_pol.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    nn.fit(E_pol)
    distances, indices = nn.kneighbors(E_proy, return_distance=True)
    bi_scores = 1.0 - distances
    return distances, indices, bi_scores
