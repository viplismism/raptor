from typing import Dict, List, Set


class Node:
    """Represents a node in the hierarchical tree structure."""

    def __init__(self, text: str, index: int, children: Set[int], embeddings) -> None:
        self.text = text
        self.index = index
        self.children = children
        self.embeddings = embeddings

    def __repr__(self) -> str:
        return (
            f"Node(index={self.index}, "
            f"children={self.children}, "
            f"text={self.text[:50]!r}{'...' if len(self.text) > 50 else ''})"
        )


class Tree:
    """Represents the entire hierarchical tree structure."""

    def __init__(
        self,
        all_nodes: Dict[int, Node],
        root_nodes: List[Node],
        leaf_nodes: Dict[int, Node],
        num_layers: int,
        layer_to_nodes: Dict[int, List[Node]],
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

    def __repr__(self) -> str:
        return (
            f"Tree(num_layers={self.num_layers}, "
            f"total_nodes={len(self.all_nodes)}, "
            f"leaf_nodes={len(self.leaf_nodes)}, "
            f"root_nodes={len(self.root_nodes)})"
        )
