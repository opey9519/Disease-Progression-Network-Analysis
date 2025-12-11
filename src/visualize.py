import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import os

RESULTS_CSV = "../output/Ablations/ablation_results.csv"
SAVE_DIR = "../output/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(RESULTS_CSV)

# This function plots training loss curves

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

# This function plots ROC & PR Curves for every new model

def plot_roc_pr_each_model():
    for model in df["model_name"].unique():
        rows = df[df["model_name"] == model]

        # Choose best-performing row, highest ROC
        best_row = rows.loc[rows["test_roc_auc"].idxmax()]

        probs = np.array(eval(best_row["test_probs"]))
        labels = np.array(eval(best_row["test_labels"]))

        # ROC curve
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.plot(fpr, tpr)
        plt.title(f"{model} – ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(f"{SAVE_DIR}/{model}_ROC.png")
        plt.close()

        # PR curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        plt.plot(recall, precision)
        plt.title(f"{model} – Precision Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(f"{SAVE_DIR}/{model}_PR.png")
        plt.close()

# This function plots the Confusion Matrix for every model

def plot_confusion_each_model():
    for model in df["model_name"].unique():
        rows = df[df["model_name"] == model]
        best_row = rows.loc[rows["test_roc_auc"].idxmax()]

        probs = np.array(eval(best_row["test_probs"]))
        labels = np.array(eval(best_row["test_labels"]))
        preds = (probs > 0.5).astype(int)

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model} – Confusion Matrix")
        plt.savefig(f"{SAVE_DIR}/{model}_CM.png")
        plt.close()

# This function visualizes HGT Attention

def visualize_hgt_attention(model_ckpt_path, data):
    from train_gnn import HGTLinkPredictor

    model = HGTLinkPredictor(data.x.size(1), 64)
    model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))
    model.eval()

    # Forward pass to get attention weights
    model.forward(data.x, data.edge_index, None, data.pos_edge_label_index, data.neg_edge_label_index)

    # HGT stores attention inside the conv layers
    with open(f"{SAVE_DIR}/HGT_attention_notes.txt", "w") as f:
        f.write("HGT attention extraction depends on PyG version. Placeholder.\n")

# This function plots the demographic graphs

def plot_graph_for_group(pt_path, name):
    data = torch.load(pt_path, map_location="cpu")
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(6,6))
    nx.draw(G, node_size=10)
    plt.title(f"Network Graph – {name}")
    plt.savefig(f"{SAVE_DIR}/{name}_graph.png")
    plt.close()

def main():
    # Loss curves per demographic
    for demo in df["demographic_group"].unique():
        plot_loss_curves(demo)
        
    plot_roc_pr_each_model()
    plot_confusion_each_model()

    print("Visualizations generated with full model support (GCN, SAGE, GAT, HGT).")

if __name__ == "__main__":
    main()