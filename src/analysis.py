import networkx as nx
import pandas as pd
from pathlib import Path
from itertools import islice
from networkx.algorithms import community

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output/demographic_networks"
OUTPUT_DIR = ROOT / "output/analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_graphs():
    """Load all demographic subgroup networks (.graphml files)."""
    graphs = {}
    for file in INPUT_DIR.glob("network_*.graphml"):
        group = file.stem.replace("network_", "")
        G = nx.read_graphml(file)
        graphs[group] = G
    return graphs


def top_edges(G, n=10):
    """Return top weighted transitions."""
    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    return edges_sorted[:n]


def centrality_measures(G):
    """Compute PageRank and degree centrality."""
    pagerank = nx.pagerank(G, weight="weight")
    indeg = dict(G.in_degree(weight="weight"))
    outdeg = dict(G.out_degree(weight="weight"))
    df = pd.DataFrame({
        "PageRank": pagerank,
        "InDegree": indeg,
        "OutDegree": outdeg
    })
    df.index.name = "Diagnosis"
    return df.sort_values("PageRank", ascending=False)


def frequent_paths(G, max_length=3, top_k=10):
    """Identify common short paths (length ≤ 3)."""
    path_counts = {}
    for src in G.nodes():
        for dst in nx.single_source_shortest_path_length(G, src, cutoff=max_length):
            path = nx.shortest_path(G, src, dst)
            if 2 <= len(path) <= max_length + 1:
                path_tuple = tuple(path)
                path_counts[path_tuple] = path_counts.get(path_tuple, 0) + 1
    top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
    return top_paths[:top_k]


def detect_communities(G):
    """Detect diagnosis clusters using greedy modularity."""
    comms = community.greedy_modularity_communities(G, weight="weight")
    communities_list = []
    for i, c in enumerate(comms):
        for node in c:
            communities_list.append({"Diagnosis": node, "Community": i})
    return pd.DataFrame(communities_list)


def analyze_all():
    graphs = load_graphs()
    summary = []

    for group, G in graphs.items():
        print(f"\nAnalyzing network: {group}")

        # --- Top transitions ---
        top_trans = top_edges(G, n=15)
        pd.DataFrame(top_trans, columns=["Src", "Dst", "Weight"]).to_csv(
            OUTPUT_DIR / f"top_transitions_{group}.csv", index=False
        )

        # --- Centrality ---
        cent = centrality_measures(G)
        cent.to_csv(OUTPUT_DIR / f"centrality_{group}.csv")

        # --- Frequent paths ---
        paths = frequent_paths(G, max_length=3, top_k=15)
        pd.DataFrame(
            [{"Path": " → ".join(p[0]), "Count": p[1]} for p in paths]
        ).to_csv(OUTPUT_DIR / f"paths_{group}.csv", index=False)

        # --- Community detection ---
        comms = detect_communities(G)
        comms.to_csv(OUTPUT_DIR / f"communities_{group}.csv", index=False)

        # --- Summary (for comparison later) ---
        summary.append({
            "Group": group,
            "NumNodes": G.number_of_nodes(),
            "NumEdges": G.number_of_edges(),
            "TopTransition": f"{top_trans[0][0]} → {top_trans[0][1]}",
            "TopTransitionWeight": top_trans[0][2]
        })

    pd.DataFrame(summary).to_csv(
        OUTPUT_DIR / "summary_all_groups.csv", index=False)
    print("\nAnalysis complete! Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    analyze_all()
