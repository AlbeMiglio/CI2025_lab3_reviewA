import numpy as np
import pandas as pd
import networkx as nx
import os
import random
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
from time import perf_counter, time

# Import our optimized engine
from algorithms.network_engine import RouteOptimizer

# --- Configuration Parameters ---
NODE_COUNTS = [10, 20, 50, 100, 200, 500]
CONNECTIVITY_RATES = [.2, .5, .8, 1.0]
DISTORTION_LEVELS = [.0, .1, .5, .8]
ALLOW_NEGATIVE_WEIGHTS = [False, True]

# --- Vectorized Graph Generation ---

def synthesize_network(node_count: int, *, connectivity: float = 1.0, 
                       allow_negatives: bool = False, distortion: float = 0.0, 
                       seed: int = 42) -> tuple:
    """
    Generates graph data using vectorized NumPy operations.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Generate Coordinates
    geo_map = rng.random(size=(node_count, 2))
    
    # 2. Calculate Euclidean Distance Matrix (Vectorized)
    diff = geo_map[:, np.newaxis, :] - geo_map[np.newaxis, :, :]
    euclidean_dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # 3. Apply Base Noise/Distortion
    base_weights = rng.random((node_count, node_count))
    if allow_negatives:
        base_weights = base_weights * 2 - 1 # Scale to [-1, 1]
    
    # Combine Euclidean distance with noise
    weighted_matrix = base_weights * distortion + euclidean_dist
    
    # 4. Apply Connectivity Mask (Sparsity)
    mask = rng.random((node_count, node_count)) < connectivity
    
    # Apply mask: Non-existent edges get Infinity
    final_matrix = np.where(mask, weighted_matrix, np.inf)
    
    # 5. Clean Diagonal (Distance to self is 0)
    np.fill_diagonal(final_matrix, 0)
    
    return (final_matrix * 1_000).round(), geo_map

def audit_network_for_negatives(matrix: np.ndarray) -> bool:
    return np.any(matrix < 0)

def plot_convergence(histories, title, filename=None):
    """Plots convergence history for analysis."""
    plt.figure(figsize=(12, 6))
    for name, history in histories.items():
        plt.plot(history, label=f"{name} (Final: {history[-1]:.2f})", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Best Cost")
    plt.title(f"Convergence Comparison: {title}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if filename:
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    plt.close()

# --- Experiment Execution ---

def execute_trial(nodes, conn, distort, use_neg):
    # 1. Create Data
    adj_matrix, coords = synthesize_network(
        nodes, 
        connectivity=conn, 
        distortion=distort, 
        allow_negatives=use_neg
    )
    
    # Check actual matrix for negatives
    has_neg = audit_network_for_negatives(adj_matrix)
    
    # 2. Convert to NetworkX for A*
    ma_matrix = np.ma.masked_invalid(adj_matrix)
    nx_graph = nx.from_numpy_array(ma_matrix, create_using=nx.DiGraph)
    
    records = []
    
    # 3. Run Solvers
    
    # -- Solver A: Dijkstra --
    dijk_dists, dijk_preds = (None, None)
    if not has_neg:
        dijk_dists, dijk_preds = RouteOptimizer.compute_all_pairs(adj_matrix, method='D')

    # -- Solver B: Bellman-Ford --
    bf_dists, bf_preds = (None, None)
    cycle_detected = False
    
    if has_neg:
        try:
            bf_dists, bf_preds = RouteOptimizer.compute_all_pairs(adj_matrix, method='BF')
        except Exception:
            cycle_detected = True

    # -- Solver C: A* --
    astar_data = (None, None)
    if not has_neg:
        astar_data = RouteOptimizer.run_heuristic_search(nx_graph, coords)
    
    # 4. Formatting Results
    for s in range(nodes):
        for t in range(nodes):
            
            # Reconstruct Dijkstra path
            d_path = None
            d_val = None
            if dijk_dists is not None and not np.isinf(dijk_dists[s, t]):
                d_val = dijk_dists[s, t]
                d_path = RouteOptimizer.reconstruct_path(dijk_preds, s, t)
                
            # Reconstruct BF path
            bf_path = None
            bf_val = None
            if cycle_detected:
                bf_val = "NEGATIVE_CYCLE"
                bf_path = "NEGATIVE_CYCLE"
            elif bf_dists is not None and not np.isinf(bf_dists[s, t]):
                bf_val = bf_dists[s, t]
                bf_path = RouteOptimizer.reconstruct_path(bf_preds, s, t)

            # Get A* data
            a_path = None
            a_val = None
            if astar_data[0] is not None:
                a_val = astar_data[0].get(s, {}).get(t)
                a_path = astar_data[1].get(s, {}).get(t)

            records.append({
                "source_node": s,
                "target_node": t,
                "metric_dijkstra": d_val,
                "path_dijkstra": d_path,
                "metric_bellman": bf_val,
                "path_bellman": bf_path,
                "metric_astar": a_val,
                "path_astar": a_path
            })

    # 5. Save to Disk
    df = pd.DataFrame(records)
    file_id = f"topology_N{nodes}_C{conn}_D{distort}_Neg{use_neg}.csv"
    save_path = os.path.join("benchmarks", file_id)
    df.to_csv(save_path, index=False)
    
    return f"Completed: {file_id}"


def analyze_apsp_performance(node_counts, connectivity=0.5, runs=3):
    """
    Benchmarks execution time AND validates correctness against Dijkstra.
    """
    results = {
        "Dijkstra (SciPy)": [],
        "Bellman-Ford (SciPy)": [],
        "A* (NetworkX)": []
    }
    
    print(f"\n{'='*80}")
    print(f"APSP PERFORMANCE & CORRECTNESS ANALYSIS (Connectivity: {connectivity})")
    print(f"{'='*80}")
    # Table Header
    print(f"{'Nodes':<8} | {'Dijkstra (s)':<13} | {'BF (s)':<13} | {'A* (s)':<13} | {'Errors':<10}")
    print("-" * 75)

    for n in node_counts:
        times_d = []
        times_bf = []
        times_a = []
        errors = 0 # Counter for mismatches
        
        for i in range(runs):
            # 1. Generate Graph
            run_seed = 42 + n + i
            adj, coords = synthesize_network(
                n, 
                connectivity=connectivity, 
                allow_negatives=False, 
                distortion=0.0, 
                seed=run_seed
            )
            
            # NetworkX graph for A*
            ma_matrix = np.ma.masked_invalid(adj)
            nx_graph = nx.from_numpy_array(ma_matrix, create_using=nx.DiGraph)

            # 2. Run Dijkstra (Gold Standard)
            t0 = perf_counter()
            d_dist, _ = RouteOptimizer.compute_all_pairs(adj, method='D')
            times_d.append(perf_counter() - t0)
            
            # 3. Run Bellman-Ford
            t0 = perf_counter()
            bf_dist, _ = RouteOptimizer.compute_all_pairs(adj, method='BF')
            times_bf.append(perf_counter() - t0)
            
            # Check BF consistency (floating point tolerance)
            if not np.allclose(d_dist, bf_dist, equal_nan=True):
                errors += 1

            # 4. Run A* (Skip for huge graphs)
            if n <= 200: 
                t0 = perf_counter()
                a_data = RouteOptimizer.run_heuristic_search(nx_graph, coords)
                times_a.append(perf_counter() - t0)
                
                # Check A* consistency
                # A* returns a dict of dicts, so we convert to matrix for comparison
                # (Simplified check: just checking random node pairs would be faster)
                a_matrix = np.full((n, n), np.inf)
                for s, targets in a_data[0].items():
                    for t, cost in targets.items():
                        a_matrix[s, t] = cost
                
                # A* finds optimal paths, so it must match Dijkstra
                if not np.allclose(d_dist, a_matrix, equal_nan=True):
                    errors += 1
            else:
                times_a.append(None)

        # Averages
        avg_d = np.mean(times_d)
        avg_bf = np.mean(times_bf)
        avg_a = np.mean(times_a) if times_a[0] is not None else np.nan
        
        results["Dijkstra (SciPy)"].append(avg_d)
        results["Bellman-Ford (SciPy)"].append(avg_bf)
        results["A* (NetworkX)"].append(avg_a)
        
        a_str = f"{avg_a:.4f}" if not np.isnan(avg_a) else "Skipped"
        err_str = "PASS" if errors == 0 else f"FAIL ({errors})"
        
        print(f"{n:<8} | {avg_d:<13.4f} | {avg_bf:<13.4f} | {a_str:<13} | {err_str:<10}")

    print("-" * 75)
    return results

def plot_apsp_benchmark(results, node_counts):
    """
    Visualizes the Time vs. Size complexity of the algorithms.
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, times in results.items():
        # Filter out NaNs (skipped runs)
        valid_indices = [i for i, t in enumerate(times) if not np.isnan(t)]
        valid_nodes = [node_counts[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]
        
        if valid_times:
            plt.plot(valid_nodes, valid_times, marker='o', label=algo_name)

    plt.xlabel("Number of Nodes (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("APSP Scalability: SciPy vs NetworkX")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.yscale("log") # Log scale helps see the massive difference
    
    filename = "apsp_benchmark_results.png"
    plt.savefig(filename)
    print(f"\nBenchmark plot saved to {filename}")
    # plt.show() # Uncomment if running in Jupyter
    
    
if __name__ == "__main__":
    output_dir = "benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Run the basic All-Pairs Shortest Path experiments
    experimental_grid = product(
        NODE_COUNTS,
        CONNECTIVITY_RATES,
        DISTORTION_LEVELS,
        ALLOW_NEGATIVE_WEIGHTS,
    )

    print(f"Starting APSP parallel execution on {os.cpu_count()} cores...")
    t_start = perf_counter()
    
    """Parallel(n_jobs=-1)( # Uncomment to run all experiments
        delayed(execute_trial)(n, c, d, neg)
        for n, c, d, neg in experimental_grid
    )"""
    
    benchmark_data = analyze_apsp_performance(
        node_counts=NODE_COUNTS, 
        connectivity=0.5, 
        runs=3
    )
    
    plot_apsp_benchmark(benchmark_data, node_counts=NODE_COUNTS)
    
    t_end = perf_counter()
    print(f"All APSP trials completed in {t_end - t_start:.2f} seconds.")