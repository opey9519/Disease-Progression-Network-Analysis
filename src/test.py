# Test if added added libraries work

import torch
from torch_geometric.data import Data

print("Torch version:", torch.__version__)

edge_index = torch.tensor([[0, 1],
                           [1, 2]], dtype=torch.long)
x = torch.tensor([[1], [2], [3]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
print(data)
