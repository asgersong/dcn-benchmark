"""GNN architecture definitions for DCN fault detection.

Each module in this package defines a PyTorch nn.Module subclass implementing
a specific GNN variant. All architectures share a common interface:

- __init__(self, config: dict) — accepts a flat config dict from the YAML
  training config; no hardcoded hyperparameters.
- forward(self, data: torch_geometric.data.Data) -> torch.Tensor — returns
  per-node or per-edge logits depending on the task head.

Available architectures (to be implemented):
- gcn.py    : Graph Convolutional Network (Kipf & Welling, 2017) baseline.
- gat.py    : Graph Attention Network v2 (Brody et al., 2022).
- gin.py    : Graph Isomorphism Network (Xu et al., 2019).
- mpnn.py   : Message Passing Neural Network with edge features.
- pinn.py   : Physics-Informed GNN.
"""