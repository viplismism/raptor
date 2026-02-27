import logging
import re
from typing import Dict, List, Set

import numpy as np
import tiktoken
from scipy import spatial

from .tree.structures import Node

logger = logging.getLogger(__name__)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer, max_tokens: int, overlap: int = None
) -> List[str]:
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.

    Args:
        text: The text to be split.
        tokenizer: The tokenizer to be used for splitting the text.
        max_tokens: The maximum allowed tokens.
        overlap: Number of overlapping sentences between chunks. Defaults to ~10% of max_tokens.

    Returns:
        A list of text chunks.
    """
    if overlap is None:
        overlap = max(1, max_tokens // 10)

    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)

    # Pre-compute token counts for each sentence, paired together
    sentence_tokens = []
    for s in sentences:
        if not s.strip():
            continue
        tc = len(tokenizer.encode(" " + s))
        sentence_tokens.append((s, tc))

    chunks = []
    current_chunk = []     # list of (sentence, token_count)
    current_length = 0

    for sentence, token_count in sentence_tokens:

        if token_count > max_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(s for s, _ in current_chunk))
                current_chunk = []
                current_length = 0

            # Split long sentence by secondary delimiters
            sub_parts = re.split(r"[,;:]", sentence)
            sub_tokens = [
                (sub.strip(), len(tokenizer.encode(" " + sub.strip())))
                for sub in sub_parts if sub.strip()
            ]

            sub_chunk = []
            sub_length = 0
            for sub, stc in sub_tokens:
                if sub_length + stc > max_tokens and sub_chunk:
                    chunks.append(" ".join(s for s, _ in sub_chunk))
                    # Keep last `overlap` sentences worth of tokens
                    keep = []
                    keep_len = 0
                    for s, t in reversed(sub_chunk):
                        if keep_len + t > overlap:
                            break
                        keep.insert(0, (s, t))
                        keep_len += t
                    sub_chunk = keep
                    sub_length = keep_len

                sub_chunk.append((sub, stc))
                sub_length += stc

            if sub_chunk:
                chunks.append(" ".join(s for s, _ in sub_chunk))

        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(s for s, _ in current_chunk))
            # Keep last sentences up to `overlap` tokens for context continuity
            keep = []
            keep_len = 0
            for s, t in reversed(current_chunk):
                if keep_len + t > overlap:
                    break
                keep.insert(0, (s, t))
                keep_len += t
            current_chunk = keep
            current_length = keep_len

            current_chunk.append((sentence, token_count))
            current_length += token_count

        else:
            current_chunk.append((sentence, token_count))
            current_length += token_count

    if current_chunk:
        chunks.append(" ".join(s for s, _ in current_chunk))

    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. "
            f"Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    return np.argsort(distances)
