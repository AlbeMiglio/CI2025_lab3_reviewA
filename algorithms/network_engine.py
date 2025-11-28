import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, NegativeCycleError

class RouteOptimizer:
    
    @staticmethod
    def compute_all_pairs(adj_matrix: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Solves All-Pairs Shortest Path using optimized C-routines via SciPy.
        
        Args:
            adj_matrix: Adjacency matrix of the graph
            method: Algorithm to use ('D' for Dijkstra, 'BF' for Bellman-Ford)
            
        Returns:
            Tuple of (distance_matrix, predecessor_matrix)
            
        Raises:
            NegativeCycleError: If negative cycle detected (only for Bellman-Ford)
            ValueError: If method is not 'D' or 'BF'
        """
        if method not in ('D', 'BF'):
            raise ValueError(f"method must be 'D' or 'BF', got {method}")
        if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError(f"adj_matrix must be square, got shape {adj_matrix.shape}")
        
        # Convert to Compressed Sparse Row format for SciPy speed
        sparse_graph = csr_matrix(adj_matrix)
        
        # SciPy calculates all distances in one C-call. 
        # return_predecessors=True gives us the data to rebuild paths.
        dist_matrix, predecessors = shortest_path(
            csgraph=sparse_graph, 
            method=method, 
            directed=True, 
            return_predecessors=True
        )
        
        return dist_matrix, predecessors

    @staticmethod
    def reconstruct_path(predecessors: np.ndarray, start: int, end: int) -> list[int] | None:
        """
        Reconstructs a specific path from the predecessor matrix.
        
        Args:
            predecessors: Predecessor matrix from shortest_path
            start: Source node index
            end: Target node index
            
        Returns:
            List of node indices representing the path, or None if no path exists
        """
        if start == end:
            return [start]
            
        path = []
        curr = end
        
        # -9999 is SciPy's default "no predecessor" marker
        if predecessors[start, end] == -9999:
            return None
            
        while curr != start:
            path.append(curr)
            curr = predecessors[start, curr]
            if curr == -9999: # Path broken
                return None
                
        path.append(start)
        return path[::-1] # Reverse to get Start -> End

    @staticmethod
    def run_heuristic_search(graph: nx.DiGraph, coords: np.ndarray) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, list[int] | None]]]:
        """
        Runs A* (A-Star) using NetworkX.
        
        Args:
            graph: NetworkX directed graph
            coords: Node coordinates for heuristic calculation
            
        Returns:
            Tuple of (distances_dict, paths_dict) where each dict is {source: {target: value}}
        """
        paths = {}
        dists = {}
        
        # --- THE FIX: Heuristic Admissibility ---
        # Graph weights are: round(dist * 1000)
        # Heuristic was:     dist * 1000
        # Problem:           10.4 (H) > 10 (Graph Weight) -> Inadmissible
        # Fix:               floor(dist * 1000) -> 10 <= 10 -> Admissible
        def dist_heuristic(u, v):
            # Use floor with scaling factor to ensure H(n) <= Cost(n)
            # The 0.9999 factor handles float precision edge cases and ensures admissibility
            # as described in README: floor(distance * 0.9999) guarantees H(n) <= C(n)
            distance = np.linalg.norm(coords[u] - coords[v]) * 1000
            return np.floor(distance * 0.9999)

        for src in graph.nodes:
            paths[src] = {}
            dists[src] = {}
            for tgt in graph.nodes:
                try:
                    # NetworkX A* is Python-based but optimized
                    p = nx.astar_path(graph, src, tgt, heuristic=dist_heuristic, weight='weight')
                    cost = nx.astar_path_length(graph, src, tgt, heuristic=dist_heuristic, weight='weight')
                    paths[src][tgt] = p
                    dists[src][tgt] = cost
                except nx.NetworkXNoPath:
                    paths[src][tgt] = None
                    dists[src][tgt] = np.inf
                    
        return dists, paths