# visualize_step_f.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import os

RESULTS_CSV = "../output/StepE_Ablations/ablation_results.csv"
SAVE_DIR = "../output/StepF_visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_CSV)

# ========= 1. TRAINING LOSS CURVES =========

def plot_loss_curves(demo):
    subset = df[df["demographic_group"] == demo]

    plt.figure(figsize=(10,6))
    for _, row in subset.iterrows():
        losses = eval(row["loss_history"])
        label = f"{row['model_name']} | {row['feature_mode']} | {row['graph_mode']}"
        plt.plot(losses, label=label)

    plt.title(f"Training Loss Curves – {demo}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)
    plt.savefig(f"{SAVE_DIR}/{demo}_loss_curves.png")
    plt.close()

# ========= 2. ROC & PR CURVES =========

def plot_roc_pr(row):
    probs = np.array(eval(row["test_probs"]))
    labels = np.array(eval(row["test_labels"]))

    # ROC
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(f"{SAVE_DIR}/{row['model_name']}_ROC.png")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(labels, probs)
    plt.plot(recall, precision)
    plt.title("Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"{SAVE_DIR}/{row['model_name']}_PR.png")
    plt.close()

# ========= 3. CONFUSION MATRIX =========

def plot_confusion_matrix(row):
    probs = np.array(eval(row["test_probs"]))
    labels = np.array(eval(row["test_labels"]))
    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{SAVE_DIR}/{row['model_name']}_CM.png")
    plt.close()

# ========= 4. GAT ATTENTION WEIGHTS =========

def visualize_gat_attention(model_ckpt_path, data):
    import torch
    from train_gnn import GATLinkPredictor

    model = GATLinkPredictor(data.x.size(1), 64)
    model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
    model.eval()

    # run one forward pass to get stored attention weights
    model.encode(data.x, data.edge_index)

    att = model.last_attention[1]     # attention values
    ei = model.last_attention[0]      # edge index

    topk = torch.topk(att, 20)
    top_edges = ei[:, topk.indices]

    with open(f"{SAVE_DIR}/top_attention_edges.txt", "w") as f:
        for i in range(20):
            src = top_edges[0][i].item()
            dst = top_edges[1][i].item()
            w = att[topk.indices[i]].item()
            f.write(f"{src} → {dst}  (α={w:.4f})\n")

# ========= 5. DEMOGRAPHIC SIDE-BY-SIDE GRAPHS =========

def plot_graph_for_group(pt_path, name):
    import torch
    data = torch.load(pt_path, map_location="cpu")

    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(6,6))
    nx.draw(G, node_size=10)
    plt.title(f"Network Graph – {name}")
    plt.savefig(f"{SAVE_DIR}/{name}_graph.png")
    plt.close()


# ========= RUN EVERYTHING =========

# Generate losses & ROC/PR/confusion for the first row of results
first_row = df.iloc[0]
plot_roc_pr(first_row)
plot_confusion_matrix(first_row)

# Loss curves for each demographic group
for demo in df["demographic_group"].unique():
    plot_loss_curves(demo)

print("Step F Visualizations generated!")
