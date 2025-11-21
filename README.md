# Computational Intelligence Lab 3: Graph Optimization & Pathfinding Analysis

This project implements and benchmarks high-performance solutions for the **All-Pairs Shortest Path (APSP)** problem. It contrasts classical algorithmic complexity theory with practical implementation realities in Python, specifically analyzing the performance gap between pure Python implementations and optimized C-based libraries.

## Project Structure

```text
.
├── algorithms/
│   ├── __init__.py
│   └── network_engine.py   # Core logic: Optimized wrapper for SciPy (C) and NetworkX
├── benchmarks/                # Output CSV datasets containing path metrics
├── main.py            # Main execution script (Generation, Benchmarking, Data Export)
└── README.md               # Project documentation
```

## Requirements
This project requires Python 3.8+ and the following scientific computing libraries:

```bash
pip install numpy scipy pandas networkx matplotlib joblib
```
## Problem Definition
The goal is to compute the shortest path between every pair of nodes in generated directed graphs under varying conditions:
1. Graph Sizes (N): 10 to 1,000 nodes.
2. Connectivity: Sparse (0.2) to Fully Connected (1.0).
3. Weights: Euclidean distance modified by random distortion/noise.
4. Negative Weights: Handling graphs where edges can have negative costs (introducing the risk of negative weight cycles).

## Implemented Solutions
We implemented a centralized engine (RouteOptimizer) that routes requests to specific solvers based on the problem constraints:

### 1. Dijkstra's Algorithm
- Implementation: scipy.sparse.csgraph.shortest_path
- Optimizations: Uses Compressed Sparse Row (CSR) matrices passed to compiled C routines.
- Use Case: Non-negative weighted graphs. Acts as the "Gold Standard" for correctness.

### 2. Bellman-Ford Algorithm
- Implementation: scipy.sparse.csgraph.shortest_path (method='BF')
- Features: Capable of handling negative edge weights.
- Cycle Detection: Explicitly catches NegativeCycleError to identify and report undefined paths in the results CSV.

### 3. A* Search
- Implementation: networkx.astar_path
- Heuristic: Euclidean Distance.
- Correction: We implemented a specific fix for Admissibility. Since graph weights are rounded integers, the raw float Euclidean distance can sometimes exceed the true cost. We applied floor(distance) to the heuristic to guarantee $h(n)≤c(n)$.

## Key Optimizations

1. Vectorized Graph Generation: We replaced standard iterative loops with NumPy broadcasting. This allows the generation of distance matrices for N=1000 graphs in milliseconds by computing (coords[:, None] - coords[None, :]) in a single vectorized operation.

2. The "SciPy" Engine: By wrapping SciPy's C-based solvers, we bypass the Python Global Interpreter Lock (GIL) and object overhead for the heavy computational lifting.

3. 3. A* Heuristic Correction

We identified a critical edge case where round(distance) in graph generation created Inadmissible Heuristics (where $H(n)>TrueCost)$.
- The Fix: We implemented a heuristic using floor(distance * 0.9999).
- Result: This guarantees $H(n)≤C(n)$ even with floating-point rounding errors, restoring A*'s optimality guarantees.

## Analysis & Findings

### 1. The "Implementation Gap"
Despite A* being algorithmically "smarter" (visiting fewer nodes), the overhead of running it in pure Python (NetworkX) outweighs its algorithmic advantage compared to a brute-force Dijkstra running in C (SciPy).
- At N=200: Dijkstra (SciPy) finished in ~0.02s.
- At N=200: A* (Python) finished in ~19.00s.

### 2. Complexity Walls
Bellman-Ford demonstrated its O(N3) complexity perfectly. While fast for small graphs, execution time exploded from 1.5s (N=200) to >60s (N=500), rendering it impractical for large, dense graphs unless negative weights are strictly required.

### 3. Correctness Verification
We validated that for non-negative graphs, all three algorithms return identical path costs (within floating-point tolerance), confirming the mathematical correctness of the implementation and the A* heuristic fix.

### 4. Negative Cycles & Reachability
When Negative Weights are enabled with high connectivity (e.g., C=0.8), the Bellman-Ford algorithm frequently returns no solution.

This is correct behavior, not a bug. In dense random graphs with negative edges, the probability of forming a Negative Weight Cycle approaches 100%.


## How to Run
To reproduce the experiments, benchmarks, and generate the dataset:

```bash
python main.py
```

This script will:

- Run a scalability benchmark comparing execution times.

- Generate a performance plot image (apsp_benchmark_results.png).

- Generate CSV files in the benchmarks/ folder containing full path data for every tested configuration.
