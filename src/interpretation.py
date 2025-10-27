import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output/demographic_networks"
ANALYSIS_DIR = ROOT / "output/analysis_results"
FIGURES_DIR = ROOT / "output/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Change groups to compare from output/demographic_networks
GROUPS_TO_COMPARE = ["Male_65+", "Female_65+"]


def load_graph(group):
    return nx.read_graphml(INPUT_DIR / f"network_{group}.graphml")


def plot_network_comparison(groups):
    fig, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 6))

    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        G = load_graph(group)
        pos = nx.spring_layout(G, seed=42)

        # Node sizes by PageRank
        centrality = nx.pagerank(G, weight="weight")
        node_sizes = [5000*centrality.get(n, 0.01) for n in G.nodes()]

        # Edge widths by weight
        edge_weights = [G[u][v].get("weight", 1)/10 for u, v in G.edges()]

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_sizes, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.7)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        ax.set_title(group)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "network_comparison.png", dpi=300)
    plt.show()


def highlight_top_transitions(group, top_n=5):
    df = pd.read_csv(ANALYSIS_DIR / f"top_transitions_{group}.csv")
    print(f"\nTop {top_n} transitions for {group}:")
    print(df.head(top_n))


if __name__ == "__main__":
    # Compare networks side by side
    plot_network_comparison(GROUPS_TO_COMPARE)

    # Print top transitions for interpretation
    for group in GROUPS_TO_COMPARE:
        highlight_top_transitions(group)
