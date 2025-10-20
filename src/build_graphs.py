import pandas as pd
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from pathlib import Path

'''
    Build a directed, weighted NetworkX graph from a DataFrame of transitions.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['src', 'dst', 'weight'].
        min_weight (int): Minimum edge weight to include an edge in the graph.

    Returns:
        nx.DiGraph: Filtered directed graph.
'''


def build_directed_graph(df: pd.DataFrame, min_weight: int = 3):
    # Filter by weight column directly
    filtered_df = df[df["weight"] >= min_weight]

    if filtered_df.empty:
        print(f"No edges meet min_weight={min_weight} - graph will be empty.")

    print(f"Edges after min_weight={min_weight}: {len(filtered_df)}")

    # Build graph directly
    graph = nx.from_pandas_edgelist(
        filtered_df, source="src", target="dst", edge_attr="weight", create_using=nx.DiGraph()
    )

    return graph


'''
    Convert a NetworkX directed graph to a PyTorch Geometric Data object.

    Parameters:
        graph (nx.DiGraph): NetworkX directed graph.

    Returns:
        torch_geometric.data.Data: PyG Data object with edge_weight tensor.
'''


def graph_to_pyg(graph: nx.DiGraph):
    data = from_networkx(graph)
    if graph.number_of_edges() > 0:
        data.edge_weight = torch.tensor(
            [d["weight"] for _, _, d in graph.edges(data=True)],
            dtype=torch.float
        )
    return data


'''
    Main function to load transitions, build graph, and save outputs.

    Parameters:
        input_csv (str): Path to CSV containing ['src', 'dst', 'weight'].
        output_prefix (str): File path prefix for saved outputs (no extension).
        min_weight (int): Minimum edge weight to filter edges.
'''


def main(input_csv: str, output_prefix: str, min_weight: int = 3):
    print(f"Reading CSV from: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"CSV loaded: {len(df)} rows")
    print(df.head())

    # Build directed graph
    G = build_directed_graph(df, min_weight)

    # Ensure output folder exists
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    # Save as GraphML for visualization / interoperability
    nx.write_graphml(G, f"{output_prefix}.graphml")

    # Save PyG Data Object for ML workflows
    torch.save(graph_to_pyg(G), f"{output_prefix}.pt")

    print(f"Graph saved: {output_prefix}.graphml / .pt")
    print(
        f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build directed weighted network from edge data."
    )
    parser.add_argument(
        "input_csv", help="Path to CSV with columns: src, dst, weight")
    parser.add_argument(
        "output_prefix", help="Output file prefix (no extension)")
    parser.add_argument("--min_weight", type=int, default=3,
                        help="Minimum edge frequency threshold")

    args = parser.parse_args()
    main(args.input_csv, args.output_prefix, args.min_weight)
