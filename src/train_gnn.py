# train_gnn.py

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import degree, to_undirected
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn.models import Node2Vec

# -----------------------------
# Config
# -----------------------------
DEMOGRAPHIC_PT_FILES = [
    "../output/demographic_networks/network_Female_18–49.pt",
    "../output/demographic_networks/network_Female_50–64.pt",
    "../output/demographic_networks/network_Female_65+.pt",
    "../output/demographic_networks/network_Male_18–49.pt",
    "../output/demographic_networks/network_Male_50–64.pt",
    "../output/demographic_networks/network_Male_65+.pt",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output" / "StepE_Ablations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "ablation_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
HIDDEN_DIM = 64
LR = 0.01

# Node2Vec config
ENABLE_NODE2VEC = True          # <-- leave True; will fall back to random if deps missing
NODE2VEC_DIM = 32
NODE2VEC_WALK_LENGTH = 20
NODE2VEC_CONTEXT_SIZE = 10
NODE2VEC_WALKS_PER_NODE = 10
NODE2VEC_EPOCHS = 20            # shorter for speed

# Ablation configs
FEATURE_MODES = [
    "diag_only",
    "diag_plus_demo",
    "node2vec",
    "temporal_degree",
]

GRAPH_MODES: Dict[str, Dict] = {
    "temporal_directed_weighted": {
        "make_undirected": False,
        "use_edge_weight": True,
    },
    "temporal_directed_unweighted": {
        "make_undirected": False,
        "use_edge_weight": False,
    },
    "temporal_undirected_weighted": {
        "make_undirected": True,
        "use_edge_weight": True,
    },
    "temporal_undirected_unweighted": {
        "make_undirected": True,
        "use_edge_weight": False,
    },
}

SCALE_FRACTIONS = [0.25, 0.5, 0.75, 1.0]


# ============================================================
# MODELS
# ============================================================

class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.link_pred = torch.nn.Linear(hidden_channels * 2, 1)

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        zcat = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(zcat).squeeze(-1)

    def forward(self, x, edge_index, edge_weight, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index, edge_weight)
        return self.decode(z, pos_edge_index), self.decode(z, neg_edge_index)


class GraphSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.link_pred = torch.nn.Linear(hidden_channels * 2, 1)

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        zcat = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(zcat).squeeze(-1)

    def forward(self, x, edge_index, edge_weight, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, pos_edge_index), self.decode(z, neg_edge_index)


class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.link_pred = torch.nn.Linear(hidden_channels * 2, 1)

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.link_pred(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)

    def forward(self, x, edge_index, edge_weight, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, pos_edge_index), self.decode(z, neg_edge_index)


# ============================================================
# FEATURE HELPERS
# ============================================================

def parse_demo_group(name: str) -> Tuple[str, str]:
    name = name.replace("network_", "")
    gender = "Female" if "Female" in name else ("Male" if "Male" in name else "Unknown")
    if "18" in name and "49" in name:
        age = "18-49"
    elif "50" in name and "64" in name:
        age = "50-64"
    elif "65" in name:
        age = "65+"
    else:
        age = "Unknown"
    return gender, age


def build_diag_only_features(n: int) -> torch.Tensor:
    return torch.eye(n, device=DEVICE)


def build_demo_features(n: int, demo_name: str) -> torch.Tensor:
    gender, age = parse_demo_group(demo_name)
    X = torch.zeros((n, 5), device=DEVICE)
    if gender == "Female":
        X[:, 0] = 1
    elif gender == "Male":
        X[:, 1] = 1
    if age == "18-49":
        X[:, 2] = 1
    elif age == "50-64":
        X[:, 3] = 1
    elif age == "65+":
        X[:, 4] = 1
    return X


def build_temporal_degree_features(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    src, dst = edge_index
    out_deg = degree(src, num_nodes=n)
    in_deg = degree(dst, num_nodes=n)
    out_deg = (out_deg - out_deg.min()) / (out_deg.max() - out_deg.min() + 1e-8)
    in_deg = (in_deg - in_deg.min()) / (in_deg.max() - in_deg.min() + 1e-8)
    return torch.stack([in_deg, out_deg], dim=1).to(DEVICE)


# ============================================================
# NODE2VEC (SAFE FALLBACK)
# ============================================================

def compute_node2vec_embeddings(data: Data) -> torch.Tensor:
    """
    Compute Node2Vec OR fallback to random if dependency missing or disabled.
    """
    if not ENABLE_NODE2VEC:
        print("Node2Vec disabled. Using random embeddings.")
        return torch.randn((data.num_nodes, NODE2VEC_DIM), device=DEVICE)

    try:
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        model = Node2Vec(
            edge_index,
            embedding_dim=NODE2VEC_DIM,
            walk_length=NODE2VEC_WALK_LENGTH,
            context_size=NODE2VEC_CONTEXT_SIZE,
            walks_per_node=NODE2VEC_WALKS_PER_NODE,
            sparse=True,
        ).to(DEVICE)

        loader = model.loader(batch_size=128, shuffle=True)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(NODE2VEC_EPOCHS):
            for batch in loader:
                optimizer.zero_grad()
                loss = model.loss(batch.to(DEVICE))
                loss.backward()
                optimizer.step()

        model.eval()
        return model().detach()

    except ImportError:
        print("WARNING: Node2Vec backend missing (torch-cluster/pyg-lib).")
        print("Using RANDOM embeddings as fallback.")
        return torch.randn((data.num_nodes, NODE2VEC_DIM), device=DEVICE)


def build_node_features(
    data: Data,
    demo_name: str,
    mode: str,
    node2vec_cache: Optional[torch.Tensor] = None,
) -> Data:
    """
    Return a new Data object with .x set according to the feature mode.
    """
    data = data.clone()

    if mode == "diag_only":
        data.x = build_diag_only_features(data.num_nodes)

    elif mode == "diag_plus_demo":
        diag = build_diag_only_features(data.num_nodes)
        demo = build_demo_features(data.num_nodes, demo_name)
        data.x = torch.cat([diag, demo], dim=1)

    elif mode == "node2vec":
        if node2vec_cache is None:
            raise RuntimeError("Node2Vec selected but embeddings not provided.")
        data.x = node2vec_cache.to(DEVICE)

    elif mode == "temporal_degree":
        temp = build_temporal_degree_features(data.edge_index, data.num_nodes)
        data.x = temp

    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    return data


# ============================================================
# GRAPH STRUCTURE & SCALE HELPERS
# ============================================================

def get_edge_weight_if_available(data: Data) -> Optional[torch.Tensor]:
    ew = getattr(data, "edge_weight", None)
    if ew is None:
        return None
    if torch.is_tensor(ew):
        return ew
    return None


def build_graph_variant(base_data: Data, config: Dict) -> Data:
    """
    Apply graph-level changes:
    - directed vs undirected
    - weighted vs unweighted
    """
    make_undirected = config.get("make_undirected", False)
    use_edge_weight = config.get("use_edge_weight", False)

    data = base_data.clone()
    edge_weight = get_edge_weight_if_available(data)

    if make_undirected:
        if edge_weight is not None:
            ei, ew = to_undirected(
                data.edge_index,
                edge_attr=edge_weight,
                num_nodes=data.num_nodes,
                reduce="add",
            )
            data.edge_index = ei
            data.edge_weight = ew
        else:
            data.edge_index = to_undirected(
                data.edge_index, num_nodes=data.num_nodes
            )

    if not use_edge_weight and hasattr(data, "edge_weight"):
        delattr(data, "edge_weight")

    return data


def subsample_edges(data: Data, frac: float, seed: int = 42) -> Data:
    """
    Subsample a fraction of edges as a proxy for using fewer patients.
    Keeps all nodes but drops some edges.
    """
    if frac >= 0.999:
        return data.clone()

    torch.manual_seed(seed)
    n_edges = data.edge_index.size(1)
    k = max(1, int(frac * n_edges))

    perm = torch.randperm(n_edges, device=data.edge_index.device)
    idx = perm[:k]

    new_data = data.clone()
    new_data.edge_index = data.edge_index[:, idx]

    edge_weight = get_edge_weight_if_available(data)
    if edge_weight is not None:
        new_data.edge_weight = edge_weight[idx]

    return new_data


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_epoch(model, data_split: Data, use_edge_weight: bool) -> float:
    model.train()
    optimizer = data_split.optimizer
    optimizer.zero_grad()
    edge_weight = get_edge_weight_if_available(data_split) if use_edge_weight else None
    pos_out, neg_out = model(
        data_split.x,
        data_split.edge_index,
        edge_weight,
        data_split.pos_edge_label_index,
        data_split.neg_edge_label_index,
    )
    out = torch.cat([pos_out, neg_out])
    label = torch.cat(
        [
            torch.ones(pos_out.size(0), device=DEVICE),
            torch.zeros(neg_out.size(0), device=DEVICE),
        ]
    )
    loss = F.binary_cross_entropy_with_logits(out, label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_split(
    model, data_split: Data, use_edge_weight: bool
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    edge_weight = get_edge_weight_if_available(data_split) if use_edge_weight else None
    pos_out, neg_out = model(
        data_split.x,
        data_split.edge_index,
        edge_weight,
        data_split.pos_edge_label_index,
        data_split.neg_edge_label_index,
    )

    out = torch.cat([pos_out, neg_out])
    label = torch.cat(
        [
            torch.ones(pos_out.size(0), device=DEVICE),
            torch.zeros(neg_out.size(0), device=DEVICE),
        ]
    )
    prob = torch.sigmoid(out).cpu()
    label_cpu = label.cpu()

    roc = roc_auc_score(label_cpu.numpy(), prob.numpy())
    pr = average_precision_score(label_cpu.numpy(), prob.numpy())

    ks = [1, 3, 5]
    sorted_idx = torch.argsort(prob, descending=True)
    hits_at_k: Dict[int, float] = {}
    for k in ks:
        k = min(k, prob.numel())
        topk_labels = label_cpu[sorted_idx[:k]]
        hits_at_k[k] = topk_labels.float().mean().item()

    return roc, pr, hits_at_k


def run_training_single_setting(
    model_class,
    model_name: str,
    demo_name: str,
    graph_mode_name: str,
    feature_mode: str,
    frac: float,
    train_data: Data,
    val_data: Data,
    test_data: Data,
    use_edge_weight: bool,
) -> Dict:
    """
    Train a single model for a given combination and return metrics.
    """
    in_channels = train_data.x.size(1)
    model = model_class(in_channels, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_data.optimizer = optimizer

    best_val_roc = -math.inf
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_data, use_edge_weight)

        if epoch % 5 == 0 or epoch == EPOCHS:
            val_roc, val_pr, _ = evaluate_split(model, val_data, use_edge_weight)
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_state_dict = model.state_dict()

            print(
                f"[{demo_name}] frac={frac}, graph={graph_mode_name}, "
                f"feat={feature_mode}, model={model_name}, "
                f"epoch={epoch:03d}, loss={loss:.4f}, val ROC={val_roc:.4f}, val PR={val_pr:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_roc, test_pr, hits_at_k = evaluate_split(model, test_data, use_edge_weight)

    ckpt_name = (
        f"{demo_name}_{model_name}_graph-{graph_mode_name}_feat-{feature_mode}_frac-{frac}.pt"
    ).replace("/", "-")
    ckpt_path = OUTPUT_DIR / ckpt_name
    torch.save(model.state_dict(), ckpt_path)

    result = {
        "demographic_group": demo_name,
        "scale_fraction": frac,
        "graph_mode": graph_mode_name,
        "feature_mode": feature_mode,
        "model_name": model_name,
        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "hits_at_1": hits_at_k.get(1, float("nan")),
        "hits_at_3": hits_at_k.get(3, float("nan")),
        "hits_at_5": hits_at_k.get(5, float("nan")),
        "checkpoint": str(ckpt_path),
    }

    print(
        f">>> TEST [{demo_name}] frac={frac}, graph={graph_mode_name}, "
        f"feat={feature_mode}, model={model_name}: "
        f"ROC={test_roc:.4f}, PR={test_pr:.4f}, "
        f"Hits@1={result['hits_at_1']:.4f}, Hits@3={result['hits_at_3']:.4f}, "
        f"Hits@5={result['hits_at_5']:.4f}"
    )

    return result


# ============================================================
# MAIN ABLATION LOOP
# ============================================================

def main():
    all_results: List[Dict] = []

    for pt_file in DEMOGRAPHIC_PT_FILES:
        demo_name = Path(pt_file).stem
        print("\n====================")
        print(f"Processing {demo_name}")
        print("====================")

        # Safe load with PyG globals
        safe_globals_list = [Data]
        extra_globals = ["DataEdgeAttr", "DataTensorAttr", "GlobalStorage"]
        for g in extra_globals:
            try:
                cls = getattr(torch_geometric.data.data, g)
                safe_globals_list.append(cls)
            except AttributeError:
                continue

        with torch.serialization.safe_globals(safe_globals_list):
            base_data: Data = torch.load(pt_file)

        print(f"Loaded: {pt_file}")
        print(f"Nodes: {base_data.num_nodes}")
        print(f"Edges: {base_data.edge_index.shape[1]}")

        base_data = base_data.to(DEVICE)

        # Node2Vec embeddings (or fallback) once per demographic graph
        print("Computing Node2Vec-like embeddings for this demographic group...")
        node2vec_z = compute_node2vec_embeddings(base_data)
        print("Embeddings ready.")

        for frac in SCALE_FRACTIONS:
            print(f"\n--- Scale fraction: {frac} ---")
            scaled_data = subsample_edges(base_data, frac)

            for graph_mode_name, graph_cfg in GRAPH_MODES.items():
                print(f"Graph mode: {graph_mode_name}")
                graph_data = build_graph_variant(scaled_data, graph_cfg)

                is_undirected = graph_cfg.get("make_undirected", False)
                transform = RandomLinkSplit(
                    is_undirected=is_undirected,
                    split_labels=True,
                    add_negative_train_samples=True,
                    num_val=0.1,
                    num_test=0.1,
                )
                train_data, val_data, test_data = transform(graph_data)

                for feat_mode in FEATURE_MODES:
                    print(f"Feature mode: {feat_mode}")

                    if feat_mode == "node2vec":
                        train_feat = train_data.clone()
                        val_feat = val_data.clone()
                        test_feat = test_data.clone()
                        train_feat.x = node2vec_z.to(DEVICE)
                        val_feat.x = node2vec_z.to(DEVICE)
                        test_feat.x = node2vec_z.to(DEVICE)
                    else:
                        train_feat = build_node_features(train_data, demo_name, feat_mode, node2vec_cache=node2vec_z)
                        val_feat = build_node_features(val_data, demo_name, feat_mode, node2vec_cache=node2vec_z)
                        test_feat = build_node_features(test_data, demo_name, feat_mode, node2vec_cache=node2vec_z)

                    use_edge_weight = graph_cfg.get("use_edge_weight", False)

                    for model_class, model_name in [
                        (GCNLinkPredictor, "GCN"),
                        (GraphSAGELinkPredictor, "GraphSAGE"),
                        (GATLinkPredictor, "GAT"),
                    ]:
                        result = run_training_single_setting(
                            model_class=model_class,
                            model_name=model_name,
                            demo_name=demo_name,
                            graph_mode_name=graph_mode_name,
                            feature_mode=feat_mode,
                            frac=frac,
                            train_data=train_feat,
                            val_data=val_feat,
                            test_data=test_feat,
                            use_edge_weight=use_edge_weight,
                        )
                        all_results.append(result)

    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll ablation results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
