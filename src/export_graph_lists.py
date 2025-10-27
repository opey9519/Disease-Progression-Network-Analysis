import networkx as nx
from pathlib import Path
import pandas as pd

# ---------- Configuration ----------
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output/demographic_networks"
OUTPUT_DIR = ROOT / "output/graph_lists"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Process ----------
for file in INPUT_DIR.glob("network_*.graphml"):
    group_name = file.stem.replace("network_", "")
    G = nx.read_graphml(file)

    # Save node list
    nodes_df = pd.DataFrame({"node": list(G.nodes())})
    nodes_df.to_csv(OUTPUT_DIR / f"nodes_{group_name}.csv", index=False)

    # Save edge list with weights
    edges_df = pd.DataFrame(
        [(u, v, d.get("weight", 1)) for u, v, d in G.edges(data=True)],
        columns=["src", "dst", "weight"]
    )
    edges_df.to_csv(OUTPUT_DIR / f"edges_{group_name}.csv", index=False)

    print(f"Exported nodes and edges for {group_name}: "
          f"{len(nodes_df)} nodes, {len(edges_df)} edges")

print(f"\nAll node/edge lists saved to {OUTPUT_DIR}")
