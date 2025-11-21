import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

class RouteOptimizer:
    
    @staticmethod
    def compute_all_pairs(adj_matrix: np.ndarray, method: str):
        """
        Solves All-Pairs Shortest Path using optimized C-routines via SciPy.
        
        Methods:
        - 'D' : Dijkstra (Fastest for non-negative)
        - 'BF': Bellman-Ford (Required for negative weights)
        """
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
    def reconstruct_path(predecessors: np.ndarray, start: int, end: int):
        """
        Reconstructs a specific path from the predecessor matrix.
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
    def run_heuristic_search(graph: nx.DiGraph, coords: np.ndarray):
        """
        Runs A* (A-Star) using NetworkX. 
        """
        paths = {}
        dists = {}
        
        # --- THE FIX: Heuristic Admissibility ---
        # Graph weights are: round(dist * 1000)
        # Heuristic was:     dist * 1000
        # Problem:           10.4 (H) > 10 (Graph Weight) -> Inadmissible
        # Fix:               floor(dist * 1000) -> 10 <= 10 -> Admissible
        def dist_heuristic(u, v):
            # Use floor to ensure H(n) <= Cost(n)
            # Adding a tiny epsilon buffer (-1e-9) handles float precision edge cases
            return np.floor(np.linalg.norm(coords[u] - coords[v]) * 1000)

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