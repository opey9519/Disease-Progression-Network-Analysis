# train_gnn.py
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# -----------------------------
# Config
# -----------------------------
DEMOGRAPHIC_PT_FILES = [
    "../output/demographic_networks/network_Female_18–49.pt",
    "../output/demographic_networks/network_Female_50–64.pt",
    "../output/demographic_networks/network_Female_65+.pt",
    "../output/demographic_networks/network_Male_18–49.pt",
    "../output/demographic_networks/network_Male_50–64.pt",
    "../output/demographic_networks/network_Male_65+.pt"
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "StepDGNN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
HIDDEN_DIM = 64
LR = 0.01

# -----------------------------
# GNN Models
# -----------------------------


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.link_pred = torch.nn.Linear(hidden_channels*2, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_cat = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(z_cat).squeeze(-1)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_out = self.decode(z, pos_edge_index)
        neg_out = self.decode(z, neg_edge_index)
        return pos_out, neg_out


class GraphSAGELinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.link_pred = torch.nn.Linear(hidden_channels*2, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_cat = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(z_cat).squeeze(-1)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_out = self.decode(z, pos_edge_index)
        neg_out = self.decode(z, neg_edge_index)
        return pos_out, neg_out


class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads=1)
        self.link_pred = torch.nn.Linear(hidden_channels*2, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        z_cat = torch.cat([z[src], z[dst]], dim=1)
        return self.link_pred(z_cat).squeeze(-1)

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encode(x, edge_index)
        pos_out = self.decode(z, pos_edge_index)
        neg_out = self.decode(z, neg_edge_index)
        return pos_out, neg_out

# -----------------------------
# Training & Evaluation
# -----------------------------


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    pos_out, neg_out = model(
        data.x,
        data.edge_index,  # full training graph edges
        data.pos_edge_label_index,  # positive edges
        data.neg_edge_label_index   # negative edges
    )
    out = torch.cat([pos_out, neg_out])
    label = torch.cat([torch.ones(pos_out.size(0), device=DEVICE),
                       torch.zeros(neg_out.size(0), device=DEVICE)])
    loss = F.binary_cross_entropy_with_logits(out, label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    pos_out, neg_out = model(
        data.x,
        data.edge_index,
        data.pos_edge_label_index,  # you could also use val/test splits
        data.neg_edge_label_index
    )
    out = torch.cat([pos_out, neg_out])
    label = torch.cat([torch.ones(pos_out.size(0), device=DEVICE),
                       torch.zeros(neg_out.size(0), device=DEVICE)])
    pred = torch.sigmoid(out)
    roc = roc_auc_score(label.cpu(), pred.cpu())
    pr = average_precision_score(label.cpu(), pred.cpu())
    return roc, pr


def run_training(model_class, data, name, demo_name):
    print(f"\nTraining {name} for {demo_name}...")
    model = model_class(data.x.size(1), HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS+1):
        loss = train(model, data, optimizer)
        if epoch % 5 == 0:
            roc, pr = test(model, data)
            print(
                f"Epoch {epoch:02d}, Loss: {loss:.4f}, ROC: {roc:.4f}, PR: {pr:.4f}")
    model_path = OUTPUT_DIR / f"{demo_name}_{name.lower()}_link_predictor.pt"
    torch.save(model.state_dict(), model_path)
    print(f"{name} model saved to {model_path}\n")

# -----------------------------
# Main
# -----------------------------


def main():
    for pt_file in DEMOGRAPHIC_PT_FILES:
        demo_name = Path(pt_file).stem
        print(f"=== Processing {demo_name} ===")

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
            data = torch.load(pt_file)

        print(f"Loaded: {pt_file}")
        print(f"Nodes: {data.num_nodes}")
        print(f"Edges: {data.edge_index.shape}")

        data = data.to(DEVICE)

        # Edge split using RandomLinkSplit
        transform = RandomLinkSplit(is_undirected=True,
                                    split_labels=True,
                                    add_negative_train_samples=True,
                                    num_val=0.1, num_test=0.1)
        train_data, val_data, test_data = transform(data)

        # Assign attributes
        train_data.train_pos_edge_index = train_data.edge_index
        train_data.train_neg_edge_index = train_data.neg_edge_label_index
        train_data.test_pos_edge_index = test_data.edge_index
        train_data.test_neg_edge_index = test_data.neg_edge_label_index

        # Node features
        if not hasattr(train_data, "x") or train_data.x is None:
            train_data.x = torch.eye(train_data.num_nodes, device=DEVICE)

        # Train models
        run_training(GCNLinkPredictor, train_data, "GCN", demo_name)
        run_training(GraphSAGELinkPredictor,
                     train_data, "GraphSAGE", demo_name)
        run_training(GATLinkPredictor, train_data, "GAT", demo_name)


if __name__ == "__main__":
    main()
