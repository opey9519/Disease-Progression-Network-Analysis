import torch
from torch_geometric.data import Data

# Try loading with weights_only=False
data = torch.load("output/network_stageB.pt", weights_only=False)
print(data)
print("Nodes:", data.num_nodes)
