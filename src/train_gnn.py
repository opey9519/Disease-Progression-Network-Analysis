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
# extra credit HGT
from torch_geometric.nn import HGTConv



# Config
DEMOGRAPHIC_PT_FILES = [
    "../output/demographic_networks/network_Female_18–49.pt",
    "../output/demographic_networks/network_Female_50–64.pt",
    "../output/demographic_networks/network_Female_65+.pt",
    "../output/demographic_networks/network_Male_18–49.pt",
    "../output/demographic_networks/network_Male_50–64.pt",
    "../output/demographic_networks/network_Male_65+.pt",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output" / "Ablations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "ablation_results.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
HIDDEN_DIM = 64
LR = 0.01

# Node2Vec config
ENABLE_NODE2VEC = True
NODE2VEC_DIM = 32
NODE2VEC_WALK_LENGTH = 20
NODE2VEC_CONTEXT_SIZE = 10
NODE2VEC_WALKS_PER_NODE = 10
NODE2VEC_EPOCHS = 20

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


# Models

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
        self.last_attention = None

    def encode(self, x, edge_index, edge_weight=None):
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = x.relu()
        x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
        self.last_attention = att2  # store attention
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.link_pred(torch.cat([z[src], z[dst]], dim=1)).squeeze(-1)

    def forward(self, x, edge_index, edge_weight, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, pos_edge_index), self.decode(z, neg_edge_index)
    
# extra credit HGT model

class HGTLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super().__init__()

        # Metadata for heterogeneous graph
        self.node_types = ["dx"]
        self.edge_types = [("dx", "progresses_to", "dx")]

        metadata = (self.node_types, self.edge_types)

        # HGT layers
        self.conv1 = HGTConv(
            in_channels,
            hidden_channels,
            metadata,
            heads,
        )

        self.conv2 = HGTConv(
            hidden_channels,
            hidden_channels,
            metadata,
            heads,
        )

        self.link_pred = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index, edge_weight, pos_edge_index, neg_edge_index):
        # Convert to heterogeneous dictionary form
        x_dict = {"dx": x}
        edge_dict = {("dx", "progresses_to", "dx"): edge_index}

        h1 = self.conv1(x_dict, edge_dict)
        h2 = self.conv2(h1, edge_dict)
        z = h2["dx"]

        pos = self.decode(z, pos_edge_index)
        neg = self.decode(z, neg_edge_index)
        return pos, neg

    def decode(self, z, edge_index):
        src, dst = edge_index
        return self.link_pred(
            torch.cat([z[src], z[dst]], dim=1)
        ).squeeze(-1)



# Feature helpers

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


# Node2vec

def compute_node2vec_embeddings(data: Data) -> torch.Tensor:
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
        print("WARNING: Node2Vec backend missing. Using random embeddings.")
        return torch.randn((data.num_nodes, NODE2VEC_DIM), device=DEVICE)


def build_node_features(
    data: Data,
    demo_name: str,
    mode: str,
    node2vec_cache: Optional[torch.Tensor] = None,
) -> Data:
    data = data.clone()

    if mode == "diag_only":
        data.x = build_diag_only_features(data.num_nodes)

    elif mode == "diag_plus_demo":
        diag = build_diag_only_features(data.num_nodes)
        demo = build_demo_features(data.num_nodes, demo_name)
        data.x = torch.cat([diag, demo], dim=1)

    elif mode == "node2vec":
        if node2vec_cache is None:
            raise RuntimeError("Node2Vec selected but embeddings missing.")
        data.x = node2vec_cache.to(DEVICE)

    elif mode == "temporal_degree":
        data.x = build_temporal_degree_features(data.edge_index, data.num_nodes)

    return data


# Graph Structure & Scale Helpers

def get_edge_weight_if_available(data: Data) -> Optional[torch.Tensor]:
    ew = getattr(data, "edge_weight", None)
    if torch.is_tensor(ew):
        return ew
    return None


def build_graph_variant(base_data: Data, config: Dict) -> Data:
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
            data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    if not use_edge_weight and hasattr(data, "edge_weight"):
        delattr(data, "edge_weight")

    return data


def subsample_edges(data: Data, frac: float, seed: int = 42) -> Data:
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


# Training & Evaluation

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
def evaluate_split(model, data_split: Data, use_edge_weight: bool):
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
    labels = torch.cat(
        [
            torch.ones(pos_out.size(0), device=DEVICE),
            torch.zeros(neg_out.size(0), device=DEVICE),
        ]
    ).cpu()

    probs = torch.sigmoid(out).cpu()

    roc = roc_auc_score(labels.numpy(), probs.numpy())
    pr = average_precision_score(labels.numpy(), probs.numpy())

    ks = [1, 3, 5]
    sorted_idx = torch.argsort(probs, descending=True)
    hits_at_k = {
        k: labels[sorted_idx[:k]].float().mean().item()
        for k in ks
    }

    return roc, pr, hits_at_k, probs.numpy(), labels.numpy()


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
):
    in_channels = train_data.x.size(1)
    model = model_class(in_channels, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_data.optimizer = optimizer

    best_val_roc = -math.inf
    best_state_dict = None
    loss_history = []  # <-- STORE LOSS

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_data, use_edge_weight)
        loss_history.append(loss)  # <-- SAVE LOSS EACH EPOCH

        if epoch % 5 == 0 or epoch == EPOCHS:
            val_roc, val_pr, _, _, _ = evaluate_split(model, val_data, use_edge_weight)
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                best_state_dict = model.state_dict()

            print(
                f"[{demo_name}] frac={frac}, graph={graph_mode_name}, "
                f"feat={feature_mode}, model={model_name}, "
                f"epoch={epoch:03d}, loss={loss:.4f}, val ROC={val_roc:.4f}"
            )

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    test_roc, test_pr, hits_at_k, probs, labels = evaluate_split(
        model, test_data, use_edge_weight
    )

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
        "hits_at_1": hits_at_k[1],
        "hits_at_3": hits_at_k[3],
        "hits_at_5": hits_at_k[5],
        "checkpoint": str(ckpt_path),
        "loss_history": loss_history,
        "test_probs": probs.tolist(),
        "test_labels": labels.tolist(),
    }

    print(
        f">>> TEST [{demo_name}] frac={frac}, graph={graph_mode_name}, "
        f"feat={feature_mode}, model={model_name}: "
        f"ROC={test_roc:.4f}"
    )

    return result


# Main Abilation Loop

def main():
    all_results = []

    for pt_file in DEMOGRAPHIC_PT_FILES:
        demo_name = Path(pt_file).stem
        print("\n====================")
        print(f"Processing {demo_name}")
        print("====================")

        safe_globals = [Data]
        for g in ["DataEdgeAttr", "DataTensorAttr", "GlobalStorage"]:
            try:
                safe_globals.append(getattr(torch_geometric.data.data, g))
            except AttributeError:
                pass

        with torch.serialization.safe_globals(safe_globals):
            base_data: Data = torch.load(pt_file)

        base_data = base_data.to(DEVICE)

        print("Computing Node2Vec embeddings...")
        node2vec_z = compute_node2vec_embeddings(base_data)
        print("Node2Vec ready.")

        for frac in SCALE_FRACTIONS:
            scaled = subsample_edges(base_data, frac)

            for graph_mode, cfg in GRAPH_MODES.items():
                graph_data = build_graph_variant(scaled, cfg)

                split = RandomLinkSplit(
                    is_undirected=cfg.get("make_undirected", False),
                    split_labels=True,
                    add_negative_train_samples=True,
                    num_val=0.1,
                    num_test=0.1,
                )

                train_data, val_data, test_data = split(graph_data)

                for feat_mode in FEATURE_MODES:

                    if feat_mode == "node2vec":
                        train_feat = train_data.clone(); train_feat.x = node2vec_z
                        val_feat = val_data.clone(); val_feat.x = node2vec_z
                        test_feat = test_data.clone(); test_feat.x = node2vec_z
                    else:
                        train_feat = build_node_features(train_data, demo_name, feat_mode, node2vec_cache=node2vec_z)
                        val_feat = build_node_features(val_data, demo_name, feat_mode, node2vec_cache=node2vec_z)
                        test_feat = build_node_features(test_data, demo_name, feat_mode, node2vec_cache=node2vec_z)

                    use_wt = cfg.get("use_edge_weight", False)

                    for model_class, model_name in [
                        (GCNLinkPredictor, "GCN"),
                        (GraphSAGELinkPredictor, "GraphSAGE"),
                        (GATLinkPredictor, "GAT"),
                        # HGT
                        (HGTLinkPredictor, "HGT"),
                    ]:
                        result = run_training_single_setting(
                            model_class,
                            model_name,
                            demo_name,
                            graph_mode,
                            feat_mode,
                            frac,
                            train_feat,
                            val_feat,
                            test_feat,
                            use_wt,
                        )
                        all_results.append(result)

    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
